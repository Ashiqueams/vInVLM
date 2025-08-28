# clip_demo.py  —  Run CLIP locally on simple, generated images 

import os
import torch
from PIL import Image, ImageDraw
import clip


def make_red_cube(path="red_cube.png"):
    """Generate a simple 'red cube' (red square) on white background."""
    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([48, 48, 208, 208], fill=(200, 0, 0))  # red square as cube proxy
    img.save(path)


def make_blue_sphere(path="blue_sphere.png"):
    """Generate a simple 'blue sphere' (blue circle) on white background."""
    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([48, 48, 208, 208], fill=(0, 80, 220))   # blue circle as sphere proxy
    img.save(path)


def ensure_test_images():
    """Create local test images if missing."""
    if not os.path.exists("red_cube.png"):
        make_red_cube("red_cube.png")
    if not os.path.exists("blue_sphere.png"):
        make_blue_sphere("blue_sphere.png")


def pick_device():
    """Pick best available device: CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def run_clip_on_image(image_path, prompts):
    """Load image, run CLIP with prompts, return sorted (prompt, score) list."""
    device = pick_device()
    print("Using device:", device)

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to open '{image_path}': {e}")

    text_tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    ranked = sorted(zip(prompts, probs), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    # 1) Ensure local sample images exist
    ensure_test_images()

    # 2) Choose which image to test
    #    Change to "blue_sphere.png" to try the other one.
    image_path = "blue_sphere.png"

    # 3) Prompts to test
    # prompts = ["a red cube", "a blue sphere", "a green triangle", "a cat", "a car"]
    # 3) Prompts to test (zero-shot variations)
    prompts = [
        "a red cube", "a crimson cube", "a ruby square",
        "a blue sphere", "a navy ball", "a blue circle",
        "a green triangle", "an emerald triangle",
        "the red shape", "the blue shape", "the green shape"
    ]

    # 4) Run CLIP
    ranked = run_clip_on_image(image_path, prompts)

    # 5) Show results
    print(f"\nImage: {image_path}")
    for p, s in ranked:
        print(f"{p:20s}  {s:.4f}")

    best = ranked[0]
    print(f"\nCLIP best guess: '{best[0]}' (score {best[1]:.4f})")


if __name__ == "__main__":
    main()