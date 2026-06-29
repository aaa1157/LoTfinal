from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final_results"
TABLES = RESULTS / "tables"
FIGURES = Path(__file__).resolve().parents[2] / "final_paper" / "figures"
CN_FONT = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=8.5)
EN_FONT = font_manager.FontProperties(fname="C:/Windows/Fonts/times.ttf", size=8.5)


STRATEGY_CN = {
    "no_metasurface": "无超表面",
    "periodic": "周期翻转",
    "random": "随机翻转",
    "rfnoid_like": "类RFNOID",
    "multifreq_proposed": "多频子阵列",
}

MODEL_CN = {
    "GradientBoosting": "梯度提升",
    "LogisticRegression": "逻辑回归",
    "RandomForest": "随机森林",
    "SVM_RBF": "RBF支持向量机",
    "PhaseCNN": "PhaseCNN",
    "ResNet1DLite": "轻量残差网络",
    "DualBranchNet": "双分支网络",
    "TinyTCN": "TinyTCN",
}

PALETTE = {
    "blue": "#1B5E9D",
    "orange": "#C47A00",
    "green": "#2F6B3E",
    "red": "#8A2D2D",
    "gray": "#666666",
    "light_blue": "#1B5E9D",
    "light_orange": "#C47A00",
}


def configure_style() -> None:
    for font_path in [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 320
    plt.rcParams["axes.edgecolor"] = "#111111"
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["xtick.labelsize"] = 8.5
    plt.rcParams["ytick.labelsize"] = 8.5
    plt.rcParams["legend.fontsize"] = 8.5
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3


def apply_axis_fonts(ax, x_font=None, y_font=EN_FONT):
    for label in ax.get_yticklabels():
        label.set_fontproperties(y_font)
    for label in ax.get_xticklabels():
        text = label.get_text()
        if x_font is not None:
            label.set_fontproperties(x_font)
        elif text and all(ord(ch) < 128 for ch in text):
            label.set_fontproperties(EN_FONT)
        else:
            label.set_fontproperties(CN_FONT)
    ax.yaxis.label.set_fontproperties(CN_FONT)
    ax.xaxis.label.set_fontproperties(CN_FONT)


def clean_axis(ax, ylim=(0.45, 1.02), chance=0.5):
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    if chance is not None:
        ax.axhline(chance, color=PALETTE["red"], linestyle=(0, (4, 3)), linewidth=0.85)


def annotate_bars(ax, bars, fmt="{:.3f}", dy=0.012):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontproperties=EN_FONT,
        )


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def make_statistical_threat_figure():
    df = pd.read_csv(TABLES / "statistical_attack_aligned_summary.csv")
    df = df[df["split_type"] == "random"].copy()
    orders = ["no_metasurface", "periodic", "random", "rfnoid_like", "multifreq_proposed"]

    def best_acc(experiment, train_setting, strategy):
        sub = df[
            (df["experiment"] == experiment)
            & (df["train_setting"] == train_setting)
            & (df["test_strategy"] == strategy)
        ]
        return float(sub["mean_accuracy"].max())

    zero_shot = [best_acc("B", "no_metasurface", s) if s != "no_metasurface" else best_acc("A", "no_metasurface", s) for s in orders]
    mixed = [best_acc("C", "mixed", s) for s in orders]

    x = np.arange(len(orders))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.3, 3.7))
    bars1 = ax.bar(x - width / 2, zero_shot, width, label="只用无超表面训练", color=PALETTE["blue"], edgecolor="#111111", linewidth=0.45, alpha=0.92)
    bars2 = ax.bar(x + width / 2, mixed, width, label="混合防护训练", color=PALETTE["orange"], edgecolor="#111111", linewidth=0.45, alpha=0.92)
    clean_axis(ax)
    ax.set_ylabel("运动识别准确率")
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_CN[s] for s in orders])
    ax.legend(frameon=True, ncol=2, loc="upper right", framealpha=1, edgecolor="#BDBDBD", prop=CN_FONT)
    annotate_bars(ax, bars1, dy=0.008)
    annotate_bars(ax, bars2, dy=0.008)
    apply_axis_fonts(ax)
    save(fig, "cn_statistical_attacker_threat_models.png")


def make_fair_comparison_figure():
    df = pd.read_csv(TABLES / "statistical_vs_deep_fair_comparison_fixed.csv")
    df = df[(df["split_type"] == "random") & (df["experiment"] == "C")].copy()
    orders = ["no_metasurface", "periodic", "random", "rfnoid_like", "multifreq_proposed"]
    df = df.set_index("test_strategy").loc[orders].reset_index()

    stat = df["best_statistical_accuracy"].astype(float).to_numpy()
    deep = df["best_deep_accuracy"].astype(float).to_numpy()
    x = np.arange(len(orders))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    bars1 = ax.bar(x - width / 2, stat, width, label="统计特征模型", color=PALETTE["blue"], edgecolor="#111111", linewidth=0.45, alpha=0.92)
    bars2 = ax.bar(x + width / 2, deep, width, label="深度时序模型", color=PALETTE["orange"], edgecolor="#111111", linewidth=0.45, alpha=0.92)
    clean_axis(ax)
    ax.set_ylabel("运动识别准确率")
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_CN[s] for s in orders])
    ax.legend(frameon=True, ncol=2, loc="upper right", framealpha=1, edgecolor="#BDBDBD", prop=CN_FONT)
    annotate_bars(ax, bars1, dy=0.008)
    annotate_bars(ax, bars2, dy=0.008)
    apply_axis_fonts(ax)
    save(fig, "cn_statistical_vs_deep_fair_comparison.png")


def make_zero_shot_deep_figure():
    df = pd.read_csv(TABLES / "statistical_vs_deep_fair_comparison_fixed.csv")
    df = df[(df["split_type"] == "random") & (df["experiment"] == "E")].copy()
    orders = ["no_metasurface", "periodic", "random", "rfnoid_like", "multifreq_proposed"]
    df = df.set_index("test_strategy").loc[orders].reset_index()

    stat = df["best_statistical_accuracy"].astype(float).to_numpy()
    deep = df["best_deep_accuracy"].astype(float).to_numpy()
    x = np.arange(len(orders))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    bars1 = ax.bar(x - width / 2, stat, width, label="统计特征模型", color=PALETTE["blue"], edgecolor="#111111", linewidth=0.45, alpha=0.92)
    bars2 = ax.bar(x + width / 2, deep, width, label="深度时序模型", color=PALETTE["orange"], edgecolor="#111111", linewidth=0.45, alpha=0.92)
    clean_axis(ax)
    ax.set_ylabel("运动识别准确率")
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_CN[s] for s in orders])
    ax.legend(frameon=True, ncol=2, loc="upper right", framealpha=1, edgecolor="#BDBDBD", prop=CN_FONT)
    annotate_bars(ax, bars1, dy=0.008)
    annotate_bars(ax, bars2, dy=0.008)
    apply_axis_fonts(ax)
    save(fig, "cn_zero_shot_statistical_vs_deep.png")


def make_lmc_figure():
    df = pd.read_csv(TABLES / "lmc_adaptive_attacker_summary.csv")
    df = df[df["split_type"] == "random"].copy()
    df["模型"] = df["model"].map(MODEL_CN)
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    bars = ax.bar(df["模型"], df["mean_accuracy"], color=colors, alpha=0.94, width=0.5, edgecolor="#111111", linewidth=0.45)
    clean_axis(ax, ylim=(0.45, 0.86), chance=0.5)
    ax.set_ylabel("运动识别准确率")
    annotate_bars(ax, bars, dy=0.008)
    apply_axis_fonts(ax)
    save(fig, "cn_lmc_adaptive_attacker_comparison.png")


def make_fingerprint_accuracy_figure():
    df = pd.read_csv(TABLES / "strategy_fingerprint_summary.csv")
    df = df[df["split_type"] == "random"].copy()
    handcrafted = float(df[df["input_type"] == "handcrafted_features"]["mean_accuracy"].max())
    raw_models = df[df["input_type"] == "raw_phase"].copy()
    rows = [
        ("最佳手工特征", handcrafted, 0.0, PALETTE["gray"]),
    ]
    for model in ["PhaseCNN", "TinyTCN", "ResNet1DLite"]:
        row = raw_models[raw_models["model_name"] == model].iloc[0]
        rows.append((MODEL_CN[model], float(row["mean_accuracy"]), float(row["std_accuracy"]), PALETTE["orange"] if model == "ResNet1DLite" else PALETTE["blue"]))

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    errors = [r[2] for r in rows]
    colors = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(6.1, 3.6))
    bars = ax.bar(labels, values, color=colors, alpha=0.94, width=0.5, edgecolor="#111111", linewidth=0.45)
    clean_axis(ax, ylim=(0.15, 0.95), chance=0.2)
    ax.set_ylabel("策略识别准确率")
    annotate_bars(ax, bars, dy=0.008)
    apply_axis_fonts(ax)
    save(fig, "cn_strategy_fingerprint_accuracy.png")


def make_fingerprint_confusion_figure():
    cm = pd.read_csv(TABLES / "strategy_fingerprint_confusion_matrix.csv", index_col=0)
    order = ["no_metasurface", "periodic", "random", "rfnoid_like", "multifreq_proposed"]
    cm = cm.loc[order, order]
    row_sum = cm.sum(axis=1).replace(0, np.nan)
    pct = cm.div(row_sum, axis=0) * 100

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=100)
    labels = [STRATEGY_CN[s] for s in order]
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测策略")
    ax.set_ylabel("真实策略")
    for i in range(len(order)):
        for j in range(len(order)):
            value = pct.iloc[i, j]
            color = "white" if value >= 55 else "#1F3A5B"
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", color=color, fontsize=9, fontproperties=EN_FONT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("行归一化比例")
    cbar.ax.yaxis.label.set_fontproperties(CN_FONT)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(EN_FONT)
    apply_axis_fonts(ax, y_font=CN_FONT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "cn_strategy_fingerprint_confusion_matrix.png")


def main():
    configure_style()
    make_statistical_threat_figure()
    make_fair_comparison_figure()
    make_zero_shot_deep_figure()
    make_lmc_figure()
    make_fingerprint_accuracy_figure()
    make_fingerprint_confusion_figure()
    print("Chinese report figures written to", FIGURES)


if __name__ == "__main__":
    main()
