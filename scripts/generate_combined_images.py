from pathlib import Path

from PIL import Image, ImageOps, ImageDraw


def build_collage(title: str, image_paths: list[Path], out_path: Path) -> None:
    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input images for {title}: {missing}")

    cols, rows = 3, 2
    tile_w, tile_h = 900, 500
    margin = 28
    top_pad = 90
    canvas_w = cols * tile_w + (cols + 1) * margin
    canvas_h = rows * tile_h + (rows + 1) * margin + top_pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill="black")

    for i, p in enumerate(image_paths):
        with Image.open(p) as img:
            tile = ImageOps.contain(img.convert("RGB"), (tile_w, tile_h), method=Image.Resampling.LANCZOS)
        bg = Image.new("RGB", (tile_w, tile_h), "white")
        xoff = (tile_w - tile.width) // 2
        yoff = (tile_h - tile.height) // 2
        bg.paste(tile, (xoff, yoff))

        col = i % cols
        row = i // cols
        x = margin + col * (tile_w + margin)
        y = top_pad + margin + row * (tile_h + margin)
        canvas.paste(bg, (x, y))

    canvas.save(out_path, format="PNG")


def main() -> None:
    fig_dir = Path("data/figures")

    core = [
        fig_dir / "figure1_architecture.png",
        fig_dir / "figure2_nar.png",
        fig_dir / "figure3_hallucination.png",
        fig_dir / "figure4_scatter.png",
        fig_dir / "figure5_gate.png",
        fig_dir / "figure6_correlation.png",
    ]
    advanced = [
        fig_dir / "figure7_loo_learning_curve.png",
        fig_dir / "figure8_metric_correlation.png",
        fig_dir / "figure9_error_composition.png",
        fig_dir / "figure10_system_profile.png",
        fig_dir / "figure11_ablation_safety_delta.png",
        fig_dir / "figure12_ablation_metric_deltas.png",
    ]

    out_1 = fig_dir / "combined_core_results.png"
    out_2 = fig_dir / "combined_advanced_results.png"

    build_collage("Core Evaluation", core, out_1)
    build_collage("Advanced + Ablation", advanced, out_2)

    print(f"created: {out_1}")
    print(f"created: {out_2}")


if __name__ == "__main__":
    main()
