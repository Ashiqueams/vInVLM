# day4_agent.py — Day 4: Simple Integration (Tiny agent uses CLIP to choose an object)

import os
from typing import List, Tuple, Dict
import torch
from PIL import Image, ImageDraw, ImageFont
import clip

# -------------------------
# Utilities
# -------------------------

def pick_device():
    """Pick best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------
# Grid World (simple 2D)
# -------------------------

COLORS: Dict[str, Tuple[int,int,int]] = {
    "red":   (200, 0, 0),
    "blue":  (0, 80, 220),
    "green": (0, 160, 90),
    "yellow":(235, 190, 0),
}

SHAPES = ["cube", "sphere", "triangle"]  # "cube"≈square proxy


def draw_shape(img: Image.Image, shape: str, color: Tuple[int,int,int], margin: int = 18):
    """
    Draw a simple shape centered in img.
    - cube   => square
    - sphere => circle
    - triangle => equilateral-ish triangle
    """
    W, H = img.size
    d = ImageDraw.Draw(img)
    left, top = margin, margin
    right, bottom = W - margin, H - margin

    if shape == "cube":
        d.rectangle([left, top, right, bottom], fill=color)
    elif shape == "sphere":
        d.ellipse([left, top, right, bottom], fill=color)
    elif shape == "triangle":
        # roughly equilateral triangle
        p1 = (W // 2, top)             # top
        p2 = (right, bottom)           # bottom-right
        p3 = (left, bottom)            # bottom-left
        d.polygon([p1, p2, p3], fill=color)
    else:
        # fallback: small dot
        cx, cy = W // 2, H // 2
        r = min(W, H) // 10
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


class GridWorld:
    """
    A tiny grid (rows x cols). Each cell may contain one (color, shape).
    We render:
    - a full grid image
    - per-cell crops (for CLIP scoring)
    """
    def __init__(
        self,
        rows: int = 1,
        cols: int = 3,
        objects: List[Tuple[str, str]] = None,
        cell_size: int = 192,
        bg_color: Tuple[int,int,int] = (255, 255, 255)
    ):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.bg_color = bg_color

        # Default layout: [("red","cube"), ("blue","sphere"), ("green","triangle")]
        if objects is None:
            objects = [("red", "cube"), ("blue", "sphere"), ("green", "triangle")]
        assert len(objects) <= rows * cols, "Too many objects for the grid"

        # Fill grid row-major; empty cells are None
        grid = [None] * (rows * cols)
        for i, obj in enumerate(objects):
            grid[i] = obj
        self.grid = grid  # list of (color, shape) or None

    def render(self) -> Tuple[Image.Image, List[Image.Image], List[Tuple[int,int,int,int]]]:
        """
        Returns:
        - full_img: PIL image of the whole grid
        - crops: list of per-cell images (same size as cell)
        - boxes: list of bounding boxes (x0,y0,x1,y1) for each cell in full_img
        """
        W = self.cols * self.cell_size
        H = self.rows * self.cell_size
        full = Image.new("RGB", (W, H), self.bg_color)
        crops: List[Image.Image] = []
        boxes: List[Tuple[int,int,int,int]] = []

        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                x0 = c * self.cell_size
                y0 = r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                boxes.append((x0, y0, x1, y1))

                # Create a cell image
                cell = Image.new("RGB", (self.cell_size, self.cell_size), "white")

                obj = self.grid[idx]
                if obj is not None:
                    color_name, shape = obj
                    color_rgb = COLORS[color_name]
                    draw_shape(cell, shape, color_rgb)

                # Paste into full
                full.paste(cell, (x0, y0))
                crops.append(cell)

        # draw grid lines
        d = ImageDraw.Draw(full)
        for r in range(1, self.rows):
            y = r * self.cell_size
            d.line([(0, y), (W, y)], fill=(220, 220, 220), width=2)
        for c in range(1, self.cols):
            x = c * self.cell_size
            d.line([(x, 0), (x, H)], fill=(220, 220, 220), width=2)

        return full, crops, boxes

    def highlight(self, full_img: Image.Image, box: Tuple[int,int,int,int], color=(0, 180, 90), width=6):
        d = ImageDraw.Draw(full_img)
        d.rectangle(box, outline=color, width=width)


# -------------------------
# CLIP scoring
# -------------------------

def cosine_similarity(model, preprocess, device, pil_img: Image.Image, instruction: str) -> float:
    """
    Compute cosine similarity between this image and the instruction text using CLIP encoders.
    (No softmax across labels; just a single prompt similarity.)
    """
    with torch.no_grad():
        image = preprocess(pil_img).unsqueeze(0).to(device)
        text = clip.tokenize([instruction]).to(device)

        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sim = (img_feat @ txt_feat.T).item()  # scalar
    return sim


def sanitize_instruction(instruction: str) -> str:
    """
    Light cleanup to improve matching.
    e.g., 'pick the red cube' -> 'a red cube'
    """
    s = instruction.strip().lower()
    s = s.replace("pick the ", "a ")
    s = s.replace("pick ", "a ")
    return s


# -------------------------
# Demo / Agent loop
# -------------------------

def run_demo(instruction: str = "pick the red cube", save_dir: str = "out"):
    ensure_dir(save_dir)
    device = pick_device()
    print("Using device:", device)

    # Load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Build a tiny world with 3 cells (left→right)
    world = GridWorld(
        rows=1,
        cols=3,
        objects=[("red", "cube"), ("blue", "sphere"), ("green", "triangle")]  # feel free to change
    )

    # Render full grid + per-cell crops
    full, crops, boxes = world.render()
    full.save(os.path.join(save_dir, "grid.png"))

    # Prepare instruction
    instr = sanitize_instruction(instruction)
    print("Instruction:", instruction, "→ Using prompt:", repr(instr))

    # Score each object vs the instruction
    scores = []
    for i, crop in enumerate(crops):
        sim = cosine_similarity(model, preprocess, device, crop, instr)
        scores.append(sim)

    # Choose best cell
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_box = boxes[best_idx]

    # Visualize: highlight chosen cell
    full_annot = full.copy()
    world.highlight(full_annot, best_box, color=(0, 180, 90), width=8)

    # Add labels under each cell (color/shape)
    draw = ImageDraw.Draw(full_annot)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        obj = world.grid[i]
        if obj:
            color_name, shape = obj
            label = f"{color_name} {shape}\nscore={scores[i]:.3f}"
        else:
            label = f"(empty)\nscore={scores[i]:.3f}"
        # text position
        draw.text((x0 + 8, y1 - 40), label, fill=(30, 30, 30))

    out_path = os.path.join(save_dir, "grid_selected.png")
    full_annot.save(out_path)

    # Print a neat summary
    print("\nScores (higher is better):")
    for i, s in enumerate(scores):
        print(f"  cell {i}: {world.grid[i]}  ->  {s:.4f}")
    print(f"\nAgent picks cell {best_idx}: {world.grid[best_idx]}")
    print(f"Saved images:\n  - {os.path.abspath(os.path.join(save_dir, 'grid.png'))}\n  - {os.path.abspath(out_path)}")


if __name__ == "__main__":
    # Try other instructions like: "pick the blue sphere", "choose the green triangle"
    run_demo("pick the red cube")