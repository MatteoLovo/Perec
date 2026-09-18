"""
Generatore di dataset per puzzle-RL (jigsaw).

Per ogni puzzle esporta:
  - i pezzi come PNG RGBA 128x128 (ritaglio sul bounding box, sfondo
    trasparente: la maschera diventa il canale alpha)
  - meta.json con ground truth, tipi dei lati, adiacenze vere, stato
    iniziale mescolato e info sullo spazio delle azioni:
    tutto cio' che serve per ricompensa e valutazione.
    Nessun tensore: la conversione a tensori e' lasciata al training loop.

Struttura output:
  dataset/
    index.json                       elenco puzzle (ordinato per difficolta')
    puzzle_000/
      meta.json
      pieces/piece_00.png ...
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configurazione
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent        # radice del progetto
IMAGES = [ROOT / "image.png"]                        # immagini sorgente
OUT_DIR = Path(__file__).resolve().parent / "dataset"

GRIDS = [(2, 2), (3, 3), (5, 6)]   # curriculum: dal piu' facile al finale
SEEDS = range(3)                   # varianti: seed diversi = incavi diversi
PIECE_SIZE = 128                   # lato del canvas quadrato di ogni pezzo
DELTA = 3                          # gradi tra i vertici dell'arco di ellisse


# --------------------------------------------------------------------------
# 1. Generazione geometrica dei pezzi
# --------------------------------------------------------------------------
def edge_points(p0, p1, kind):
    """Polilinea da p0 a p1. kind: 'flat' | 'convex' | 'concave'.

    Il rigonfiamento e' una mezza ellisse (cv2.ellipse2Poly). Percorrendo il
    contorno in senso orario (coordinate immagine, y verso il basso) la
    normale esterna e' sempre -n, quindi:
      convesso = arco 180->360,  concavo = arco 0->180 (invertito).
    """
    if kind == "flat":
        return [tuple(p0), tuple(p1)]

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    length = np.hypot(*d)
    d = d / length
    center = (p0 + p1) / 2
    axes = (int(round(length * 0.20)), int(round(length * 0.25)))
    angle = int(round(np.degrees(np.arctan2(d[1], d[0]))))
    c_int = tuple(int(v) for v in np.round(center))

    if kind == "convex":
        arc = cv2.ellipse2Poly(c_int, axes, angle, 180, 360, DELTA)
    else:  # concave
        arc = cv2.ellipse2Poly(c_int, axes, angle, 0, 180, DELTA)[::-1]
    return [tuple(p0.astype(int))] + [tuple(p) for p in arc] + [tuple(p1.astype(int))]


def piece_edges(r, c, rows, cols, v_edges, h_edges):
    """Tipi dei 4 lati del pezzo (r, c): piatto sul bordo, altrimenti
    convesso/concavo in modo complementare al vicino."""
    top = "flat" if r == 0 else ("convex" if not h_edges[r - 1][c] else "concave")
    bottom = "flat" if r == rows - 1 else ("convex" if h_edges[r][c] else "concave")
    left = "flat" if c == 0 else ("convex" if not v_edges[r][c - 1] else "concave")
    right = "flat" if c == cols - 1 else ("convex" if v_edges[r][c] else "concave")
    return {"top": top, "right": right, "bottom": bottom, "left": left}


def to_square_rgba(bgr, mask, size):
    """Ritaglia sul bounding box della maschera, applica l'alpha e
    ricampiona su un canvas quadrato size x size (sfondo trasparente)."""
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
    piece = cv2.cvtColor(bgr[y:y + h, x:x + w], cv2.COLOR_BGR2BGRA)
    piece[:, :, 3] = mask[y:y + h, x:x + w]

    side = max(w, h)
    canvas = np.zeros((side, side, 4), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = piece
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA), [x, y, w, h]


# --------------------------------------------------------------------------
# 2. Generazione di un puzzle completo
# --------------------------------------------------------------------------
def generate_puzzle(img, rows, cols, seed, out_dir):
    """Genera i pezzi RGBA e il meta.json di un singolo puzzle."""
    h_img, w_img = img.shape[:2]
    rng = random.Random(seed)

    xs = [round(c * w_img / cols) for c in range(cols + 1)]
    ys = [round(r * h_img / rows) for r in range(rows + 1)]
    v_edges = [[rng.random() < 0.5 for _ in range(cols - 1)] for _ in range(rows)]
    h_edges = [[rng.random() < 0.5 for _ in range(cols)] for _ in range(rows - 1)]

    pieces_dir = out_dir / "pieces"
    pieces_dir.mkdir(parents=True, exist_ok=True)

    pieces = []
    for r in range(rows):
        for c in range(cols):
            pid = r * cols + c
            x0, x1 = xs[c], xs[c + 1]
            y0, y1 = ys[r], ys[r + 1]
            tl, tr, br, bl = (x0, y0), (x1, y0), (x1, y1), (x0, y1)
            edges = piece_edges(r, c, rows, cols, v_edges, h_edges)

            contour = (
                edge_points(tl, tr, edges["top"])
                + edge_points(tr, br, edges["right"])[1:]
                + edge_points(br, bl, edges["bottom"])[1:]
                + edge_points(bl, tl, edges["left"])[1:]
            )
            contour = np.array(contour, dtype=np.int32)

            mask = np.zeros((h_img, w_img), np.uint8)
            cv2.fillPoly(mask, [contour], 255)

            rgba, bbox = to_square_rgba(img, mask, PIECE_SIZE)
            cv2.imwrite(str(pieces_dir / f"piece_{pid:02d}.png"), rgba)

            pieces.append({
                "id": pid,
                "file": f"pieces/piece_{pid:02d}.png",
                "gt_row": r, "gt_col": c, "gt_slot": r * cols + c,
                "is_border": r in (0, rows - 1) or c in (0, cols - 1),
                "is_corner": r in (0, rows - 1) and c in (0, cols - 1),
                "edges": edges,
                "bbox": bbox,  # [x, y, w, h] nel canvas originale
            })

    # Adiacenze vere (ricompensa densa): [pezzo_a, pezzo_b, direzione]
    true_adjacencies = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                true_adjacencies.append([r * cols + c, r * cols + c + 1, "right"])
            if r + 1 < rows:
                true_adjacencies.append([r * cols + c, (r + 1) * cols + c, "bottom"])

    # 3. Stato iniziale: pezzi mescolati sugli slot (un pezzo per slot)
    n = rows * cols
    initial_board = list(range(n))
    rng.shuffle(initial_board)

    meta = {
        "grid": [rows, cols],
        "seed": seed,
        "n_pieces": n,
        "canvas": [w_img, h_img],
        "pieces": pieces,
        "true_adjacencies": true_adjacencies,
        "rl": {
            "n_slots": n,
            "n_pieces": n,
            "rotation": False,          # pezzi gia' orientati: niente rotazione
            "initial_board": initial_board,  # initial_board[slot] = piece_id
            "reward": {
                "dense": "true_adjacencies + edge-shape consistency",
                "sparse": "tutti i pezzi al proprio gt_slot",
            },
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# --------------------------------------------------------------------------
# 4-5. Caricamento e metriche (reward / valutazione per il training RL)
# --------------------------------------------------------------------------
def load_puzzle(puzzle_dir):
    """Ritorna (meta, immagini RGBA come array numpy HxWx4, indicizzate per
    piece id). Da qui ai tensori manca solo la conversione."""
    puzzle_dir = Path(puzzle_dir)
    with open(puzzle_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    imgs = [cv2.imread(str(puzzle_dir / p["file"]), cv2.IMREAD_UNCHANGED)
            for p in meta["pieces"]]
    return meta, imgs


def placement_metrics(placement, meta):
    """placement[slot] = piece_id (tutti gli slot occupati).

    Ritorna:
      exact_accuracy      frazione di pezzi nel proprio gt_slot
      adjacency_accuracy  frazione di adiacenze vere rispettate (reward densa)
      edge_score          frazione di regole geometriche rispettate
                          (piatto sul bordo, convesso contro concavo)
      solved              True se il puzzle e' completamente corretto
    """
    rows, cols = meta["grid"]
    edges = {p["id"]: p["edges"] for p in meta["pieces"]}
    gt_slot = {p["id"]: p["gt_slot"] for p in meta["pieces"]}
    slot_of = {pid: s for s, pid in enumerate(placement)}

    exact = sum(slot_of[pid] == s for pid, s in gt_slot.items()) / len(gt_slot)

    adj_ok = 0
    for a, b, direction in meta["true_adjacencies"]:
        shift = 1 if direction == "right" else cols
        adj_ok += slot_of[b] - slot_of[a] == shift
    adjacency = adj_ok / len(meta["true_adjacencies"])

    def complementary(e1, e2):
        pair = {e1, e2}
        return pair == {"convex", "concave"}

    rules_ok = rules_tot = 0
    for slot, pid in enumerate(placement):
        r, c = divmod(slot, cols)
        e = edges[pid]
        rules = [
            (e["top"] != "flat" or r == 0),
            (e["bottom"] != "flat" or r == rows - 1),
            (e["left"] != "flat" or c == 0),
            (e["right"] != "flat" or c == cols - 1),
        ]
        if c + 1 < cols:  # complementarita' con il vicino a destra
            other = edges[placement[slot + 1]]
            rules.append(complementary(e["right"], other["left"]))
        if r + 1 < rows:  # complementarita' con il vicino sotto
            other = edges[placement[slot + cols]]
            rules.append(complementary(e["bottom"], other["top"]))
        rules_ok += sum(rules)
        rules_tot += len(rules)

    return {
        "exact_accuracy": exact,
        "adjacency_accuracy": adjacency,
        "edge_score": rules_ok / rules_tot,
        "solved": exact == 1.0,
    }


# --------------------------------------------------------------------------
# Main: costruzione del dataset
# --------------------------------------------------------------------------
def main():
    index = []
    count = 0
    for img_path in IMAGES:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        for rows, cols in GRIDS:          # curriculum: griglie crescenti
            for seed in SEEDS:            # varianti: seed = incavi diversi
                pid = f"puzzle_{count:03d}"
                generate_puzzle(img, rows, cols, seed, OUT_DIR / pid)
                index.append({
                    "id": pid,
                    "dir": pid,
                    "grid": [rows, cols],
                    "n_pieces": rows * cols,
                    "seed": seed,
                    "source": img_path.name,
                })
                count += 1

    index.sort(key=lambda p: p["n_pieces"])  # curriculum: facile -> difficile
    with open(OUT_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    total_pieces = sum(p["n_pieces"] for p in index)
    print(f"Dataset: {len(index)} puzzle, {total_pieces} pezzi totali")
    for p in index:
        print(f"  {p['id']}: griglia {p['grid'][0]}x{p['grid'][1]} "
              f"({p['n_pieces']} pezzi), seed={p['seed']}")


if __name__ == "__main__":
    main()
