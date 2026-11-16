"""Publication figures (PDF + PNG) generated only from saved result files."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import LABELS, Timer, out_dir

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
AQUA, VIOLET = "#1baf7a", "#6250d6"
# Protocol identity: colour plus marker shape (aqua is below 3:1 contrast, so shape and legend carry identity too).
PROTO = {"random_window": ("Random-window CV", BLUE, "o"), "segment_grouped": ("Segment-grouped CV", AQUA, "s"),
         "participant_grouped": ("Participant-grouped CV", ORANGE, "D")}
FSETS = {"full": ("All 561 features", INK2, "o", True), "no_gravity": ("Without 60 gravity features", VIOLET, "D", False)}
SHORT = {"WALKING": "Walk", "WALKING_UPSTAIRS": "Upstairs", "WALKING_DOWNSTAIRS": "Downstairs",
         "SITTING": "Sit", "STANDING": "Stand", "LAYING": "Lay"}
MODEL_NAMES = {"LogReg": "Logistic Reg.", "SVM_RBF": "RBF SVM", "RandomForest": "Random Forest", "Dummy": "Majority"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42})
FIG = out_dir("figures")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def class_distribution(tables):
    d = pd.read_csv(tables / "class_distribution.csv")
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    x, w = np.arange(len(LABELS)), 0.38
    for i, (split, color) in enumerate([("train", BLUE), ("test", ORANGE)]):
        s = d[d.split == split].set_index("activity").reindex(LABELS)
        n = s["count"].sum()
        ax.bar(x + (i - 0.5) * w, s.proportion * 100, w - 0.03, color=color, label=f"Official {split} (n={n:,})")
    ax.set_xticks(x, [SHORT[l] for l in LABELS])
    ax.set_ylabel("Share of windows (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "class_distribution")


def repeated_protocols(rep):
    """Repeat-level OOF macro-F1 for every split seed: one dot per seed, a bar at the mean."""
    r = pd.read_csv(rep / "repeat_metrics_full.csv")
    models = ["LogReg", "SVM_RBF", "RandomForest"]
    fig, ax = plt.subplots(figsize=(5.4, 1.75))  # printed at this size in the paper
    y = np.arange(len(models))[::-1]
    offsets = dict(zip(PROTO, [0.24, 0.0, -0.24]))
    for key, (label, color, marker) in PROTO.items():
        for yy, mod in zip(y, models):
            v = r[(r.model == mod) & (r.strategy == key)].macro_f1
            ax.scatter(v, np.full(len(v), yy + offsets[key]), s=14, marker=marker, facecolor="none", edgecolor=color,
                       lw=0.9, zorder=3, label=label if mod == models[0] else None)
            ax.vlines(v.mean(), yy + offsets[key] - 0.1, yy + offsets[key] + 0.1, color=color, lw=2, zorder=4)
    ax.set_yticks(y, [MODEL_NAMES[m] for m in models])
    ax.set_xlabel(f"Pooled out-of-fold macro-F1 (marker = one split seed, bar = mean of {r.seed.nunique()})")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0), handletextpad=0.2)
    save(fig, "repeated_protocols")


def confusion_matrix(metrics):
    cm = pd.read_csv(metrics / "test_confusion_counts.csv", index_col=0).loc[LABELS, LABELS]
    pct = cm.div(cm.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(1.75, 1.45))  # printed at natural size in the paper
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = pct.iat[i, j]
            ax.text(j, i, f"{cm.iat[i, j]}", ha="center", va="center", fontsize=6,
                    color="white" if v > 55 else INK)
    ax.set_xticks(range(6), [SHORT[l] for l in LABELS], rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(6), [SHORT[l] for l in LABELS], fontsize=6.5)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("Predicted activity", fontsize=7, labelpad=1)
    ax.set_ylabel("True activity", fontsize=7, labelpad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "test_confusion_matrix")  # shading = row %, explained in the caption


def per_class_drop(tables):
    """Per-class F1 drop (random-window minus participant-grouped), full vs gravity-ablated features."""
    d = pd.read_csv(tables / "repeated_per_class.csv")
    d = d[d.comparison == "random_minus_participant"]
    models = ["LogReg", "SVM_RBF", "RandomForest"]
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.4), sharey=True, sharex=True)  # printed at this size
    y = np.arange(len(LABELS))[::-1]
    for ax, mod in zip(axes, models):
        for (fs, (label, color, marker, fill)), off in zip(FSETS.items(), [0.15, -0.15]):
            s = d[(d.model == mod) & (d.feature_set == fs)].set_index("activity").reindex(LABELS)
            ax.hlines(y + off, s["min"], s["max"], color=color, lw=0.9, zorder=2)
            ax.scatter(s["mean"], y + off, s=16, marker=marker, facecolor=color if fill else "white", edgecolor=color,
                       lw=0.9, zorder=3, label=label)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(MODEL_NAMES[mod])
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [SHORT[l] for l in LABELS])
    fig.subplots_adjust(wspace=0.12)
    fig.supxlabel("F1 gap: random-window minus participant-grouped (mean; line = min–max over seeds)", fontsize=7.5, y=-0.12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    save(fig, "per_class_drop_ablation")


def subject_activity_heatmap(tables):
    d = pd.read_csv(tables / "class_counts_by_subject.csv", index_col=0)
    d = pd.concat([d[d.split == "train"], d[d.split == "test"]])
    counts = d[LABELS].T
    fig, ax = plt.subplots(figsize=(5.5, 1.9))
    im = ax.imshow(counts, cmap="Blues", aspect="auto")
    ax.set_yticks(range(6), [SHORT[l] for l in LABELS])
    ax.set_xticks(range(len(d)), [str(s) for s in d.index], fontsize=6.3)
    n_train = int((d.split == "train").sum())
    ax.axvline(n_train - 0.5, color=INK, lw=1)
    ax.text(n_train / 2 - 0.5, -0.9, "official train (21)", ha="center", fontsize=7, color=INK2)
    ax.text(n_train + (len(d) - n_train) / 2 - 0.5, -0.9, "official test (9)", ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("Participant ID")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Windows", fontsize=7)
    cb.outline.set_visible(False)
    save(fig, "subject_activity_counts")


def per_subject_test(tables):
    d = pd.read_csv(tables / "test_per_subject.csv").sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.scatter(d.macro_f1, range(len(d)), s=22, color=BLUE)
    ax.set_yticks(range(len(d)), [f"P{s}" for s in d.subject])
    ax.set_xlabel("Test macro-F1 per participant")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "test_per_subject")


def main():
    tables, metrics, rep = out_dir("tables"), out_dir("metrics"), out_dir("repeated")
    class_distribution(tables)
    subject_activity_heatmap(tables)
    if (rep / "repeat_metrics_full.csv").exists():
        repeated_protocols(rep)
    if (tables / "repeated_per_class.csv").exists() and (rep / "repeat_metrics_no_gravity.csv").exists():
        per_class_drop(tables)
    if (metrics / "test_confusion_counts.csv").exists():
        confusion_matrix(metrics)
        per_subject_test(tables)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    with Timer("make_figures"):
        main()
