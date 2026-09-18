"""
env.py — punto di ingresso unico: ambiente (gymnasium) + training DQN
+ visualizzazione real-time del modello che gioca (trial and error).

Perche' DQN e non PPO: problema deterministico, azioni discrete e poche
transizioni "decisive" -> off-policy con replay buffer riusa ogni mossa
migliaia di volte. I logits della SwapHead [S*S] sono i Q-value degli swap.

Uso:
    python env.py                 (training headless)
    python replay.py              (guarda il modello allenato, greedy)

Curriculum (cambia PUZZLE_DIR):
    data/dataset/puzzle_000   2x2 (16 azioni / 12 swap)   <- parti da qui
    data/dataset/puzzle_003   3x3 (81 azioni)
    data/dataset/puzzle_006   5x6 (900 azioni)

Finestra cv2 (se PUZZLE_RENDER=1): ESC = esci, P = pausa.
Env: PUZZLE_STEPS, PUZZLE_RENDER, PUZZLE_BATCH.
"""

import os
import random
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from data.pieces import load_puzzle
from model.model import (PuzzleActorCritic, PuzzleReward, build_swap_mask,
                         edges_to_tensor)

torch.set_num_threads(int(os.environ.get("PUZZLE_THREADS", "4")))

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
PUZZLE_DIR = ROOT / "data" / "dataset" / "puzzle_000"   # 2x2: curriculum start
MAX_STEPS_FACTOR = 3          # max mosse per episodio = factor * n_pezzi
RENDER = os.environ.get("PUZZLE_RENDER", "0") == "1"   # headless di default
USE_VISION = False          # Fase 0: solo FORME (lati). CNN lenta su CPU:
                            # prima si valida l'RL, poi si accende la visione
FREEZE_ENCODER = True       # Fase 0: feature fisse -> DQN stabile (altrimenti
                            # l'encoder si muove sotto i Q-value: diverge)

# DQN
TOTAL_STEPS = int(os.environ.get("PUZZLE_STEPS", "20000"))
BATCH = int(os.environ.get("PUZZLE_BATCH", "128"))
BUFFER_SIZE = 20_000
GAMMA = 0.95
LR = 1e-3
EPS_START, EPS_END, EPS_ANNEAL = 1.0, 0.05, 4_000   # esplorazione decrescente
TARGET_SYNC = 100             # ogni quanti update copio la rete nel target
UPDATE_EVERY = 2              # 1 update di rete ogni 2 step: il costo e' li'
LOG_EVERY = 500               # passi di env tra un log e l'altro
EVAL_EVERY = 1_000            # passi tra una valutazione greedy e l'altra
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Visualizzatore: il board come immagine, mossa per mossa
# ---------------------------------------------------------------------------
class Renderer:
    def __init__(self, name, meta, piece_images_bgra):
        rows, cols = meta["grid"]
        self.rows, self.cols = rows, cols
        self.tile = max(64, min(160, 640 // max(rows, cols)))
        self.header = 64
        self.w, self.h = cols * self.tile, self.header + rows * self.tile
        self.name = name
        self.tiles = [cv2.resize(im, (self.tile, self.tile))
                      for im in piece_images_bgra]

    def draw(self, board, last_swap, text1, text2, solved=False):
        frame = np.full((self.h, self.w, 3), 40, np.uint8)
        for slot in range(board.numel()):
            r, c = divmod(slot, self.cols)
            x, y = c * self.tile, self.header + r * self.tile
            tile = self.tiles[int(board[slot])]
            roi = frame[y:y + self.tile, x:x + self.tile].astype(np.float32)
            alpha = tile[:, :, 3:4].astype(np.float32) / 255
            frame[y:y + self.tile, x:x + self.tile] = (
                tile[:, :, :3].astype(np.float32) * alpha + roi * (1 - alpha)
            ).astype(np.uint8)
            cv2.rectangle(frame, (x, y), (x + self.tile, y + self.tile),
                          (70, 70, 70), 1)
        if last_swap is not None:
            for slot, color in zip(last_swap, ((0, 220, 0), (0, 200, 255))):
                r, c = divmod(int(slot), self.cols)
                x, y = c * self.tile, self.header + r * self.tile
                cv2.rectangle(frame, (x, y), (x + self.tile, y + self.tile),
                              color, 3)
        cv2.putText(frame, text1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, text2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (160, 200, 255), 1, cv2.LINE_AA)
        if solved:
            cv2.rectangle(frame, (0, 0), (self.w - 1, self.h - 1),
                          (0, 255, 0), 6)
            cv2.putText(frame, "RISOLTO!", (self.w // 4, self.h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv2.imshow("Puzzle RL - training", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt
        if key == ord("p"):
            cv2.waitKey(0)


# ---------------------------------------------------------------------------
# Ambiente gymnasium: swap di due slot, reward assoluta da PuzzleReward
# ---------------------------------------------------------------------------
class PuzzleEnv(gym.Env):
    """Osservazione: board[slot] = piece_id. Azione: indice di S*S (due slot)."""

    metadata = {"render_modes": []}

    def __init__(self, puzzle_dir, render=True):
        super().__init__()
        self.meta, self.piece_images = load_puzzle(puzzle_dir)
        self.rows, self.cols = self.meta["grid"]
        self.n = self.meta["n_pieces"]
        self.max_steps = MAX_STEPS_FACTOR * self.n
        self.reward_fn = PuzzleReward(self.meta)
        self.action_space = spaces.Discrete(self.n * self.n)
        self.observation_space = spaces.MultiDiscrete([self.n] * self.n)
        self.renderer = Renderer(Path(puzzle_dir).name, self.meta,
                                 self.piece_images) if render else None
        self.episode = 0
        self.best_adj = 0.0
        self._gen = torch.Generator().manual_seed(0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._gen = torch.Generator().manual_seed(seed)
        self.board = torch.randperm(self.n, generator=self._gen)
        self.steps = 0
        self.last_swap = None
        self.episode += 1
        if self.renderer:
            self._render(0.0, self.reward_fn.scores(self.board))
        return self.board.numpy().copy(), {}

    def step(self, action):
        a, b = divmod(int(action), self.n)
        # NB: MAI fare `board[a], board[b] = board[b], board[a]` su tensori
        # torch: l'indicizzazione restituisce viste 0-dim con aliasing sulla
        # storage -> il primo assegnamento corrompe il secondo (duplicazione).
        va, vb = self.board[a].item(), self.board[b].item()
        self.board[a], self.board[b] = vb, va
        self.last_swap = (a, b)
        self.steps += 1

        reward, scores = self.reward_fn(self.board)   # reward assoluta
        terminated = bool(scores["solved"])
        truncated = self.steps >= self.max_steps
        self.best_adj = max(self.best_adj, scores["adjacency"])

        if self.renderer:
            self._render(reward, scores)
        info = {"adjacency": scores["adjacency"], "exact": scores["exact"],
                "solved": terminated}
        return self.board.numpy().copy(), reward, terminated, truncated, info

    def _render(self, reward, scores):
        text1 = (f"{self.renderer.name}  ep {self.episode}  "
                 f"passo {self.steps}/{self.max_steps}")
        text2 = (f"reward {reward:+.3f}   adiacenze {scores['adjacency']:.0%}"
                 f"   best {self.best_adj:.0%}")
        self.renderer.draw(self.board, self.last_swap, text1, text2,
                           solved=scores["solved"])

    def close(self):
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Metrica vera: success rate della policy greedy (argmax sui Q-value)
# ---------------------------------------------------------------------------
def greedy_eval(model, env, images, edges, mask, episodes=10):
    """IMPORTANTE: embeddings ricalcolati QUI dai pesi correnti (no caching):
    valutare con embedding vecchi misura un modello che non esiste."""
    with torch.no_grad():
        piece_emb = model.encode_pieces(images, edges)
    solved, adjs = 0, []
    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            board_t = torch.tensor(obs, dtype=torch.long,
                                   device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q, _ = model.forward_from_embeds(board_t, piece_emb, mask)
            obs, _, term, trunc, info = env.step(int(q.argmax(dim=-1)))
            if term or trunc:
                break
        solved += int(term)
        adjs.append(info["adjacency"])
    return solved / episodes, sum(adjs) / len(adjs)


# ---------------------------------------------------------------------------
# Training Double-DQN
# ---------------------------------------------------------------------------
def train():
    torch.manual_seed(0)
    random.seed(0)
    env = PuzzleEnv(PUZZLE_DIR, render=RENDER)
    eval_env = PuzzleEnv(PUZZLE_DIR, render=False)
    n = env.n
    model = PuzzleActorCritic(n_pieces=n, use_vision=USE_VISION).to(DEVICE)
    target = PuzzleActorCritic(n_pieces=n, use_vision=USE_VISION).to(DEVICE)
    ckpt_dir = ROOT / "model" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # RESUME: ogni run riparte dal miglior modello allenato finora
    ckpt = ckpt_dir / f"best_{PUZZLE_DIR.name}.pt"
    resumed = False
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        resumed = True
    target.load_state_dict(model.state_dict())

    if FREEZE_ENCODER:
        for p in model.fusion.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=LR)

    images = torch.stack([torch.from_numpy(im).permute(2, 0, 1).float() / 255
                          for im in env.piece_images]).unsqueeze(0).to(DEVICE)
    edges = edges_to_tensor(env.meta).unsqueeze(0).to(DEVICE)
    mask = build_swap_mask(n).unsqueeze(0).to(DEVICE)
    with torch.no_grad():  # per raccolta ed eval: embedding cached (no grad)
        piece_emb = model.encode_pieces(images, edges)
        t_emb = target.encode_pieces(images, edges)  # cached, rifatto al sync
    valid = mask[0].nonzero().flatten().tolist()

    buffer = deque(maxlen=BUFFER_SIZE)
    print(f"DQN su {PUZZLE_DIR.name} ({env.rows}x{env.cols}, {n} pezzi, "
          f"{n * n} azioni) — device {DEVICE}")

    obs, _ = env.reset()
    ep_ret, ep_returns, ep_solved = 0.0, deque(maxlen=50), deque(maxlen=50)
    grad_steps = 0
    # best = livello del modello caricato: il checkpoint si sovrascrive
    # SOLO se la run corrente lo supera davvero (mai un modello peggiore)
    best, best_adj = greedy_eval(model, eval_env, images, edges, mask)
    eps_start = 0.3 if resumed else EPS_START   # ripresa: meno esplorazione
    if resumed:
        print(f"Resume: caricato {ckpt.name} — greedy iniziale "
              f"risolti {best:.0%} | adj {best_adj:.0%} | eps {eps_start}")
    for step in range(1, TOTAL_STEPS + 1):
        # --- agisci: epsilon-greedy sui Q-value (la SwapHead e' la Q) ---
        eps = max(EPS_END, eps_start - step / EPS_ANNEAL * (eps_start - EPS_END))
        board_t = torch.tensor(obs, dtype=torch.long,
                               device=DEVICE).unsqueeze(0)
        if random.random() < eps:
            action = random.choice(valid)
        else:
            with torch.no_grad():
                q, _ = model.forward_from_embeds(board_t, piece_emb, mask)
            action = int(q.argmax(dim=-1))

        next_obs, reward, term, trunc, info = env.step(action)
        buffer.append((obs.copy(), action, reward, next_obs.copy(),
                       term or trunc))
        obs = next_obs
        ep_ret += reward
        if term or trunc:
            ep_returns.append(ep_ret)
            ep_solved.append(int(term))
            ep_ret = 0.0
            obs, _ = env.reset()

        # --- update: ogni transizione viene riusata via replay buffer ---
        if len(buffer) >= BATCH and step % UPDATE_EVERY == 0:
            batch = random.sample(buffer, BATCH)
            b0 = torch.as_tensor(np.array([t[0] for t in batch]),
                                 dtype=torch.long, device=DEVICE)
            a0 = torch.tensor([t[1] for t in batch], device=DEVICE)
            r0 = torch.tensor([t[2] for t in batch], dtype=torch.float32,
                              device=DEVICE)
            b1 = torch.as_tensor(np.array([t[3] for t in batch]),
                                 dtype=torch.long, device=DEVICE)
            d0 = torch.tensor([t[4] for t in batch], dtype=torch.float32,
                              device=DEVICE)

            # una sola forward per s e s' (concatenati): meta' del costo
            bb = torch.cat([b0, b1])
            # encoder congelato -> si riusa l'embedding cached (stazionario)
            emb = piece_emb if FREEZE_ENCODER else model.encode_pieces(
                images, edges)
            q_cat, _ = model.forward_from_embeds(
                bb, emb.expand(2 * BATCH, -1, -1), mask.expand(2 * BATCH, -1))
            q_all = q_cat[:BATCH]
            q_next_online = q_cat[BATCH:].detach()
            q_a = q_all.gather(1, a0.unsqueeze(1)).squeeze(1)

            with torch.no_grad():          # Double DQN: target (cached al sync)
                best_a = q_next_online.argmax(dim=1, keepdim=True)
                q_next, _ = target.forward_from_embeds(
                    b1, t_emb.expand(BATCH, -1, -1), mask.expand(BATCH, -1))
                q_next = q_next.gather(1, best_a).squeeze(1)
                y = r0 + GAMMA * q_next * (1 - d0)         # target Bellman

            # MSE e non Huber: con err grandi il gradiente di smooth_l1
            # satura a 1 e il bonus di risoluzione (+5) non si propaga mai
            loss = torch.nn.functional.mse_loss(q_a, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            grad_steps += 1
            if grad_steps % TARGET_SYNC == 0:
                target.load_state_dict(model.state_dict())
                with torch.no_grad():  # rinfresca gli embedding cached
                    piece_emb = model.encode_pieces(images, edges)
                    t_emb = target.encode_pieces(images, edges)

        # --- log + eval greedy + checkpoint ---
        if step % LOG_EVERY == 0 and ep_returns:
            print(f"[{step:6d}] eps {eps:.2f} | return "
                  f"{np.mean(ep_returns):+6.2f} | risolti "
                  f"{np.mean(ep_solved):4.0%} | buffer {len(buffer)}")
        if step % EVAL_EVERY == 0:
            g_solved, g_adj = greedy_eval(model, eval_env, images, edges, mask)
            if g_solved > best or not ckpt.exists():
                best = max(best, g_solved)
                torch.save(model.state_dict(), ckpt)
            print(f"        >>> greedy: risolti {g_solved:4.0%} | "
                  f"adj {g_adj:4.0%} | best {best:4.0%}")

    print(f"Fine. Best greedy: {best:.0%}. Checkpoint in {ckpt_dir}")
    env.close()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
