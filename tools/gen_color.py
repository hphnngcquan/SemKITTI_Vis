import json
import random

def generate_color_palette(num_colors=256, seed=42):
    random.seed(seed)
    palette = {}
    for i in range(num_colors):
        # Random RGB in [0,1]
        palette[i] = [random.random(), random.random(), random.random()]
    return palette

if __name__ == "__main__":
    palette = generate_color_palette(1000)  # generate 1024 colors
    with open("color_palette.json", "w") as f:
        json.dump(palette, f, indent=2)
    print("✅ Saved new color_palette.json with", len(palette), "entries")