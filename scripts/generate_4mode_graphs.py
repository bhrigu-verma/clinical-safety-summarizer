from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


COLORS = {
    "ml": "#2E86AB",
    "dl_base": "#A8DADC",
    "finetuned": "#1D3557",
    "t5xxl": "#E76F51",
}

MODE_LABELS = {
    "ml": "ML deterministic",
    "dl_base": "DL base (T5-XL)",
    "finetuned": "DL fine-tuned",
    "t5xxl": "T5-XXL",
}


def pick_latest_final_csv(eval_dir: Path) -> Path:
    files = sorted(eval_dir.glob("final_4mode_comparison_tier1_*.csv"))
    if not files:
        raise FileNotFoundError("No final_4mode_comparison_tier1_*.csv found in data/eval_results")
    return files[-1]


def ensure_modes(df: pd.DataFrame) -> pd.DataFrame:
    order = ["ml", "dl_base", "finetuned", "t5xxl"]
    out = df[df["mode"].isin(order)].copy()
    out["mode"] = pd.Categorical(out["mode"], categories=order, ordered=True)
    return out.sort_values("mode")


def annotate_bars(ax):
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h):
            ax.annotate(
                f"{h:.3f}",
                (patch.get_x() + patch.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )


def save_figure(fig: plt.Figure, out_base: Path) -> list[Path]:
    out_png = out_base.with_suffix(".png")
    out_svg = out_base.with_suffix(".svg")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", facecolor="white")
    return [out_png, out_svg]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    eval_dir = root / "data" / "eval_results"
    fig_dir = root / "data" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    src = pick_latest_final_csv(eval_dir)
    df = pd.read_csv(src)
    df = ensure_modes(df)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.labelweight": "medium",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
    })

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    created: list[Path] = []

    # Prettier mode labels
    df = df.copy()
    df["mode_label"] = df["mode"].astype(str).map(MODE_LABELS)

    # Figure A: Faithfulness metrics
    faith_cols = ["nar_mean", "hr_mean", "or_mean", "acr_mean"]
    m1 = df[["mode_label"] + faith_cols].melt(id_vars="mode_label", var_name="metric", value_name="value")
    fig = plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=m1, x="metric", y="value", hue="mode_label", palette=[COLORS[m] for m in ["ml", "dl_base", "finetuned", "t5xxl"]])
    ax.set_title("Figure 16: 4-Model Faithfulness Metrics (Tier-1)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Mode", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    created += save_figure(fig, fig_dir / f"figure16_4mode_faithfulness_{ts}")
    plt.close(fig)

    # Figure B: Safety and fluency comparison
    comp_cols = ["safety_score_mean", "fluency_score_mean"]
    m2 = df[["mode_label"] + comp_cols].melt(id_vars="mode_label", var_name="metric", value_name="value")
    fig = plt.figure(figsize=(9.5, 5.5))
    ax = sns.barplot(data=m2, x="mode_label", y="value", hue="metric", palette=["#264653", "#2a9d8f"])
    annotate_bars(ax)
    ax.set_title("Figure 17: Safety vs Fluency by Model (Tier-1)")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=12)
    created += save_figure(fig, fig_dir / f"figure17_4mode_safety_fluency_{ts}")
    plt.close(fig)

    # Figure C: Safety-Fluency frontier scatter
    fig = plt.figure(figsize=(8.5, 6.2))
    ax = plt.gca()
    for _, r in df.iterrows():
        mode = str(r["mode"])
        mode_label = MODE_LABELS[mode]
        ax.scatter(r["fluency_score_mean"], r["safety_score_mean"], s=170, color=COLORS[mode], edgecolor="black", linewidth=0.5)
        ax.text(r["fluency_score_mean"] + 0.004, r["safety_score_mean"] + 0.004, mode_label, fontsize=9)
    ax.set_title("Figure 18: 4-Model Frontier (Safety vs Fluency)")
    ax.set_xlabel("Fluency score (higher better)")
    ax.set_ylabel("Safety score (higher better)")
    ax.grid(alpha=0.35)
    created += save_figure(fig, fig_dir / f"figure18_4mode_frontier_{ts}")
    plt.close(fig)

    # Figure D: Latency comparison (log scale)
    fig = plt.figure(figsize=(8.5, 5.2))
    ax = sns.barplot(data=df, x="mode_label", y="latency_ms_mean", hue="mode_label", dodge=False, palette=[COLORS[m] for m in ["ml", "dl_base", "finetuned", "t5xxl"]], legend=False)
    annotate_bars(ax)
    ax.set_yscale("log")
    ax.set_title("Figure 19: Latency Mean by Model (Tier-1, log scale)")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Latency ms (log)")
    ax.tick_params(axis="x", rotation=12)
    created += save_figure(fig, fig_dir / f"figure19_4mode_latency_{ts}")
    plt.close(fig)

    print(f"SOURCE_CSV {src}")
    for path in created:
        print(f"FIGURE {path}")


if __name__ == "__main__":
    main()
