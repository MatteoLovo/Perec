"""
Rete e reward per il puzzle-RL (stile swap: board = permutazione dei pezzi).

Architettura (tensordict-agnostica, torch puro e testabile):
  PieceCNNEncoder   [B, N, 4, 128, 128] -> [B, N, D]   (RGBA: l'alpha e' la forma)
  EdgeEncoder       [B, N, 4, 3]        -> [B, N, D]   (one-hot lati, ramo tabellare)
  SlotEmbedding     posizione appresa per ogni slot
  SlotTransformer   self-attention tra slot (norm_first, batch_first)
  SwapHead          MLP sulla coppia -> logits [B, S*S], con maschera
  ValueHead         flatten degli slot -> scalare [B]

Reward (PuzzleReward): ASSOLUTA per stato — piccola guida densa su
adiacenze/lati + bonus dominante alla risoluzione. Coerente con
placement_metrics di data/pieces.py.
"""

import json
from pathlib import Path

import torch
from torch import nn

EDGE_TO_IDX = {"flat": 0, "convex": 1, "concave": 2}


# ---------------------------------------------------------------------------
# Layer: encoder visivo dei pezzi
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv 3x3 -> GroupNorm -> SiLU (GroupNorm: batch piccoli del RL)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PieceCNNEncoder(nn.Module):
    """[B, N, 4, H, W] -> [B, N, d_model]. Primo layer a 4 canali (RGBA)."""

    def __init__(self, d_model=128, in_channels=4):
        super().__init__()
        self.stages = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),   # 128 -> 64
            ConvBlock(32, 64, stride=2),            # 64  -> 32
            ConvBlock(64, 128, stride=2),           # 32  -> 16
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) #global aver pooling
        self.proj = nn.Linear(128, d_model) #proiezione lineare

    def forward(self, images):
        b, n = images.shape[:2]
        x = images.flatten(0, 1)                    # [B*N, 4, H, W]
        x = self.pool(self.stages(x)).flatten(1)    # [B*N, 128]
        return self.proj(x).view(b, n, -1)          # [B, N, D]


# ---------------------------------------------------------------------------
# Layer: encoder tabellare dei lati + fusione
# ---------------------------------------------------------------------------
class EdgeEncoder(nn.Module):
    """One-hot dei 4 lati [B, N, 4, 3] -> [B, N, d_model]. Ramo tabellare:
    da solo e' sufficiente per la Fase 0 (puzzle solo-geometrico)."""

    def __init__(self, d_model=128, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * 3, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, d_model),
        )

    def forward(self, edges_onehot):
        b, n = edges_onehot.shape[:2]
        return self.mlp(edges_onehot.reshape(b, n, -1))


class PieceFusion(nn.Module):
    """Somma embedding visivo + tabellare (use_vision=False per la Fase 0)."""

    def __init__(self, d_model=128, use_vision=True):
        super().__init__()
        self.use_vision = use_vision
        self.vision = PieceCNNEncoder(d_model) if use_vision else None
        self.edges = EdgeEncoder(d_model)

    def forward(self, images, edges_onehot):
        out = self.edges(edges_onehot)
        if self.use_vision:
            out = out + self.vision(images)
        return out                                  # [B, N, D]


# ---------------------------------------------------------------------------
# Layer: board -> sequenza di slot -> transformer
# ---------------------------------------------------------------------------
class SlotEmbedding(nn.Module):
    """Embedding posizionale appreso per ogni slot."""

    def __init__(self, n_slots, d_model):
        super().__init__()
        self.pos = nn.Embedding(n_slots, d_model)

    def forward(self, slot_features):               # [B, S, D]
        return slot_features + self.pos.weight.unsqueeze(0)


class SlotTransformer(nn.Module):
    """Self-attention tra slot: ogni slot vede i vicini (e tutto il board)."""

    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


# ---------------------------------------------------------------------------
# Teste
# ---------------------------------------------------------------------------
class SwapHead(nn.Module):
    """Score MLP sulla coppia (slot_a, slot_b) -> logits appiattiti [B, S*S].
    L'azione e' un singolo indice i = slot_a * S + slot_b.

    NO prodotto scalare q.k: una forma bilineare q(h_i).k(h_j) ha capacita'
    limitata (ogni logit e' vincolato a un rank-1 sulla coppia) e converge
    lentamente in fit supervisionato di Q(s,a). Qui ogni coppia concatenata
    [h_i; h_j] passa in un MLP: score arbitrario per coppia, stesso costo
    asintotico (broadcast, niente loop sulle coppie)."""

    def __init__(self, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.SiLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, h, mask=None):
        b, s, d = h.shape
        hi = h.unsqueeze(2).expand(b, s, s, d)      # feature di slot_a
        hj = h.unsqueeze(1).expand(b, s, s, d)      # feature di slot_b
        logits = self.mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)  # [B,S,S]
        logits = logits.flatten(1)                  # [B, S*S]
        if mask is not None:                        # mask: True = azione valida
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits


class ValueHead(nn.Module):
    """Flatten degli slot -> scalare [B].

    NO mean-pooling: mediare sugli slot distruggerebbe l'informazione
    "quale pezzo sta in quale slot" (la media e' quasi identica per ogni
    permutazione) e il critic diventerebbe cieco. Il flatten preserva
    la disposizione esatta: il critic puo' stimare il valore di ogni
    dei n! stati possibili."""

    def __init__(self, d_model=128, n_slots=30):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_slots * d_model, d_model), nn.SiLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, h):
        return self.mlp(h.flatten(1)).squeeze(-1)   # [B]


# ---------------------------------------------------------------------------
# Modello completo (actor + critic nello stesso modulo, encoder condiviso)
# ---------------------------------------------------------------------------
class PuzzleActorCritic(nn.Module):
    """forward(board, images, edges_onehot, mask) -> (logits [B,S*S], value [B]).

    board: [B, S] long, board[slot] = piece_id (permutazione, stile swap).
    Prima versione: ricalcola gli embedding dei pezzi a ogni step (semplice);
    l'ottimizzazione 'una volta per episodio' si aggiunge dopo.
    """

    def __init__(self, n_pieces, d_model=128, nhead=4, num_layers=2,
                 use_vision=True):
        super().__init__()
        self.n_slots = n_pieces
        self.fusion = PieceFusion(d_model, use_vision)
        self.slot_emb = SlotEmbedding(n_pieces, d_model)
        self.trunk = SlotTransformer(d_model, nhead, num_layers)
        self.policy = SwapHead(d_model)
        self.value = ValueHead(d_model, n_pieces)

    def encode_pieces(self, images, edges_onehot):
        return self.fusion(images, edges_onehot)    # [B, N, D]

    def forward_from_embeds(self, board, piece_emb, mask=None):
        """Come forward ma con embedding gia' calcolati: le immagini dei
        pezzi non cambiano mai durante un episodio, quindi si possono
        codificare una sola volta e riusare a ogni step."""
        b, s = board.shape
        d = piece_emb.shape[-1]
        slot_feat = torch.gather(
            piece_emb, 1, board.unsqueeze(-1).expand(b, s, d))     # [B, S, D]
        h = self.trunk(self.slot_emb(slot_feat))                   # [B, S, D]
        return self.policy(h, mask), self.value(h)

    def forward(self, board, images, edges_onehot, mask=None):
        piece_emb = self.encode_pieces(images, edges_onehot)       # [B, N, D]
        return self.forward_from_embeds(board, piece_emb, mask)


def build_swap_mask(n_slots):
    """Maschera base degli swap: vietato solo slot_a == slot_b. [S*S] bool."""
    eye = torch.eye(n_slots, dtype=torch.bool).flatten()
    return ~eye


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------
class PuzzleReward:
    """Reward assoluta + bonus risoluzione, calcolata da meta.json
    (mai nel modello!).

    reward(stato) = w_adj * adjacency + w_edge * edge_score
                    - step_penalty + solve_bonus se risolto.
    """

    def __init__(self, meta, adjacency_weight=0.05, edge_weight=0.02,
                 solve_bonus=5.0, step_penalty=0.01):
        # Gerarchia: la guida densa e' piccola di proposito (~0.05/passo),
        # la ricompensa vera e' RISOLVERE (bonus 5.0 terminale). Se la guida
        # pagasse troppo, la strategia ottima diventerebbe "restare in stati
        # quasi-risolti" invece di risolvere (episodio termina al solve).
        rows, cols = meta["grid"]
        n = meta["n_pieces"]
        self.w_adj, self.w_edge = adjacency_weight, edge_weight
        self.solve_bonus, self.step_penalty = solve_bonus, step_penalty

        # ground truth: gt_slot[p] = slot corretto del pezzo p
        self.gt_slot = torch.tensor([p["gt_slot"] for p in meta["pieces"]])

        # adiacenze vere: (a, b, shift) con b = a + shift nella soluzione
        adj = meta["true_adjacencies"]
        self.adj_a = torch.tensor([x[0] for x in adj])
        self.adj_b = torch.tensor([x[1] for x in adj])
        self.adj_shift = torch.tensor([1 if x[2] == "right" else cols for x in adj])

        # tabella dei 4 lati per pezzo: [N, 4] di indici EDGE_TO_IDX
        dirs = ("top", "right", "bottom", "left")
        self.piece_edges = torch.tensor(
            [[EDGE_TO_IDX[p["edges"][d]] for d in dirs] for p in meta["pieces"]])

        # regole "piatto sul bordo": border_rules[p, s] = regole rispettate (0-4)
        flat = EDGE_TO_IDX["flat"]
        border_ok = torch.ones(n, n, 4, dtype=torch.bool)  # default: regola ok
        for s in range(n):
            r, c = divmod(s, cols)
            border_ok[:, s, 0] = (self.piece_edges[:, 0] != flat) | (r == 0)
            border_ok[:, s, 3] = (self.piece_edges[:, 3] != flat) | (c == 0)
            border_ok[:, s, 1] = (self.piece_edges[:, 1] != flat) | (c == cols - 1)
            border_ok[:, s, 2] = (self.piece_edges[:, 2] != flat) | (r == rows - 1)
        self.border_count = border_ok.float().sum(-1)       # [N, S]

        # compatibilita' convesso-concavo: compat_h[p, q] per p a sx di q, ecc.
        right_e, left_e = self.piece_edges[:, 1], self.piece_edges[:, 3]
        bottom_e, top_e = self.piece_edges[:, 2], self.piece_edges[:, 0]
        self.compat_h = ((right_e[:, None] != flat) & (left_e[None, :] != flat)
                         & (right_e[:, None] != left_e[None, :]))       # [N, N]
        self.compat_v = ((bottom_e[:, None] != flat) & (top_e[None, :] != flat)
                         & (bottom_e[:, None] != top_e[None, :]))       # [N, N]

        # coppie di slot geometricamente adiacenti + denominatore regole
        self.slot_pairs_h, self.slot_pairs_v = [], []
        rules_per_slot = torch.full((n,), 4.0)
        for s in range(n):
            r, c = divmod(s, cols)
            if c + 1 < cols:
                self.slot_pairs_h.append((s, s + 1))
                rules_per_slot[s] += 1
            if r + 1 < rows:
                self.slot_pairs_v.append((s, s + cols))
                rules_per_slot[s] += 1
        self.rules_total = rules_per_slot.sum()

    def scores(self, board):
        """board: [S] long (permutazione). -> dict adjacency/edge/solved/exact."""
        slot_of = torch.empty_like(board)
        slot_of[board] = torch.arange(board.numel())

        adjacency = ((slot_of[self.adj_b] - slot_of[self.adj_a])
                     == self.adj_shift).float().mean().item()

        pieces = board                                             # piece per slot
        num = self.border_count[pieces, torch.arange(board.numel())].sum()
        if self.slot_pairs_h:
            ia, ib = map(torch.tensor, zip(*self.slot_pairs_h))
            num = num + self.compat_h[board[ia], board[ib]].sum()
        if self.slot_pairs_v:
            ja, jb = map(torch.tensor, zip(*self.slot_pairs_v))
            num = num + self.compat_v[board[ja], board[jb]].sum()
        edge = (num / self.rules_total).item()

        exact = (slot_of == self.gt_slot).float().mean().item()
        return {"adjacency": adjacency, "edge": edge,
                "exact": exact, "solved": exact == 1.0}

    def __call__(self, board):
        """Reward ASSOLUTA dello stato (non delta): ogni stato buono paga
        subito, senza dipendere dal passo precedente -> segnale denso e
        credit assignment piu' facile per PPO."""
        s = self.scores(board)
        reward = (self.w_adj * s["adjacency"] + self.w_edge * s["edge"]
                  - self.step_penalty
                  + (self.solve_bonus if s["solved"] else 0.0))
        return reward, s

    @classmethod
    def from_json(cls, meta_path, **kwargs):
        with open(meta_path, encoding="utf-8") as f:
            return cls(json.load(f), **kwargs)


# ---------------------------------------------------------------------------
# Helper: meta.json -> tensori di input per la rete
# ---------------------------------------------------------------------------
def edges_to_tensor(meta):
    """[N, 4, 3] one-hot dei lati, dai metadati."""
    n = meta["n_pieces"]
    dirs = ("top", "right", "bottom", "left")
    out = torch.zeros(n, 4, 3)
    for p in meta["pieces"]:
        for j, d in enumerate(dirs):
            out[p["id"], j, EDGE_TO_IDX[p["edges"][d]]] = 1.0
    return out


# ---------------------------------------------------------------------------
# Self-test: shape dei layer + sanita' della reward su un puzzle reale
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    root = Path(__file__).resolve().parent.parent
    meta = json.load(open(root / "data" / "dataset" / "puzzle_006" / "meta.json",
                          encoding="utf-8"))
    n = meta["n_pieces"]

    # 1. layer: shape test (batch=2)
    model = PuzzleActorCritic(n_pieces=n)
    board = torch.stack([torch.randperm(n), torch.randperm(n)])
    images = torch.rand(2, n, 4, 128, 128)
    edges = edges_to_tensor(meta).unsqueeze(0).expand(2, n, 4, 3)
    mask = build_swap_mask(n).unsqueeze(0).expand(2, -1)
    logits, value = model(board, images, edges, mask)
    assert logits.shape == (2, n * n) and value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    print(f"layer OK: logits {tuple(logits.shape)}, value {tuple(value.shape)}")

    # 2. reward: soluzione esatta -> punteggi massimi
    reward_fn = PuzzleReward(meta)
    gt_board = torch.empty(n, dtype=torch.long)
    for p in meta["pieces"]:
        gt_board[p["gt_slot"]] = p["id"]
    print("reward, soluzione:", reward_fn.scores(gt_board))

    # 3. reward assoluta: stato risolto >> stato rotto
    swapped = gt_board.clone()
    swapped[0], swapped[1] = swapped[1].item(), swapped[0].item()
    r_solved, _ = reward_fn(gt_board)
    r_broken, scores = reward_fn(swapped)
    print(f"reward risolto={r_solved:+.3f} vs rotto={r_broken:+.3f} "
          f"(scores={scores})")
