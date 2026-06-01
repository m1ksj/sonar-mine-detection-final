from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGS = ROOT / "reports" / "figures" / "final_presentation"
ANALYSIS = ROOT / "experiments" / "analysis_sources"
YOLO26N_RESULTS = ANALYSIS / "yolo26n_final_results_csv"
YOLOV4_LOGS = ANALYSIS / "yolov4_slurm_logs"
YOLOV4_PLAN = ANALYSIS / "final_training_plan.csv"
DATASET = ROOT / "data" / "processed" / "yolo26n"

FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

YOLO26N_SETTING_ORDER = [
    "YOLO26n no aug",
    "YOLO26n YOLOv4-style aug",
]

SETTING_ORDER = [
    "YOLOv4 default",
    "YOLOv4 no aug",
    "YOLO26n no aug",
    "YOLO26n YOLOv4-style aug",
]

SETTING_COLORS = {
    "YOLOv4 default": "#d62728",
    "YOLOv4 no aug": "#ff9896",
    "YOLO26n no aug": "#9ecae1",
    "YOLO26n YOLOv4-style aug": "#1f77b4",
}

CATEGORY_ORDER = ["Empty", "MILCO-only", "NOMBO-only", "Mixed"]
CATEGORY_COLORS = {
    "Empty": "#7f7f7f",
    "MILCO-only": "#d62728",
    "NOMBO-only": "#1f77b4",
    "Mixed": "#9467bd",
}

SPLIT_ORDER = ["Train", "Validation", "Test"]


def norm(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "_")
    )


def find_col(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    lookup = {norm(col): col for col in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(
            f"Could not find any of {candidates}. "
            f"Columns: {list(df.columns)}"
        )
    return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def mean_sem(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan"), float("nan")
    mean_value = float(values.mean())
    if len(values) == 1:
        sem_value = 0.0
    else:
        sem_value = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean_value, sem_value


def savefig(filename: str) -> None:
    plt.tight_layout()
    path = FIGS / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def add_setting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    model_col = find_col(df, ["model", "model_name"], required=False)
    aug_col = find_col(df, ["augmentation", "aug"], required=False)

    if model_col is None or aug_col is None:
        raise KeyError(
            "Need model and augmentation columns to derive setting names."
        )

    def map_setting(model: str, aug: str) -> str:
        model_l = str(model).lower()
        aug_l = str(aug).lower()

        if "yolov4" in model_l:
            if "no_aug" in aug_l or "no aug" in aug_l:
                return "YOLOv4 no aug"
            return "YOLOv4 default"

        if "yolo26" in model_l:
            if "no_aug" in aug_l or "no aug" in aug_l:
                return "YOLO26n no aug"
            if "yolov4" in aug_l:
                return "YOLO26n YOLOv4-style aug"
            return "YOLO26n no aug"

        return f"{model} | {aug}"

    df["setting"] = [
        map_setting(m, a)
        for m, a in zip(df[model_col], df[aug_col])
    ]
    return df


def summarize_final_runs() -> pd.DataFrame:
    df = add_setting(read_csv(TABLES / "final_run_results.csv"))

    cols = {
        "test_precision": find_col(
            df,
            ["test_precision", "precision"],
            required=False,
        ),
        "test_recall": find_col(
            df,
            ["test_recall", "recall"],
            required=False,
        ),
        "test_f1": find_col(df, ["test_f1", "f1"], required=False),
        "test_map50": find_col(
            df,
            ["test_map50", "map50"],
            required=False,
        ),
        "test_map50_95": find_col(
            df,
            ["test_map50_95", "map50_95", "map50-95"],
            required=False,
        ),
        "runtime_seconds": find_col(
            df,
            ["runtime_seconds"],
            required=False,
        ),
        "best_weights_mb": find_col(
            df,
            ["best_weights_mb"],
            required=False,
        ),
    }

    rows = []
    for setting in SETTING_ORDER:
        group = df[df["setting"] == setting]
        if group.empty:
            continue

        row = {
            "setting": setting,
            "n_runs": len(group),
        }

        for key, col in cols.items():
            if col is not None:
                row[f"{key}_mean"], row[f"{key}_sem"] = mean_sem(group[col])

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "presentation_final_run_summary.csv", index=False)
    print(f"wrote {TABLES / 'presentation_final_run_summary.csv'}")
    return summary


def plot_bar_with_sem(
    df: pd.DataFrame,
    mean_col: str,
    sem_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    plot_df = df[df["setting"].isin(SETTING_ORDER)].copy()
    plot_df["setting"] = pd.Categorical(
        plot_df["setting"],
        categories=SETTING_ORDER,
        ordered=True,
    )
    plot_df = plot_df.sort_values("setting")

    x = np.arange(len(plot_df))
    y = plot_df[mean_col].to_numpy()
    err = plot_df[sem_col].fillna(0).to_numpy()
    colors = [SETTING_COLORS[s] for s in plot_df["setting"]]

    plt.figure(figsize=(9, 5))
    plt.bar(x, y, yerr=err, capsize=5, color=colors)
    plt.xticks(x, plot_df["setting"], rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)

    savefig(filename)


def collect_dataset_audit_rows() -> pd.DataFrame:
    rows = []

    split_specs = [
        ("Train", DATASET / "images" / "train", DATASET / "labels" / "train"),
        ("Validation", DATASET / "images" / "val", DATASET / "labels" / "val"),
        (
            "Validation",
            DATASET / "images" / "valid",
            DATASET / "labels" / "valid",
        ),
        ("Test", DATASET / "images" / "test", DATASET / "labels" / "test"),
    ]

    for split, image_dir, label_dir in split_specs:
        if not image_dir.exists() or not label_dir.exists():
            continue

        for image_path in sorted(
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
            + list(image_dir.glob("*.png"))
        ):
            label_path = label_dir / f"{image_path.stem}.txt"
            classes = []

            if label_path.exists() and label_path.stat().st_size > 0:
                for line in label_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue

                    if class_id == 0:
                        classes.append("MILCO")
                    elif class_id == 1:
                        classes.append("NOMBO")

            has_milco = "MILCO" in classes
            has_nombo = "NOMBO" in classes

            if has_milco and has_nombo:
                category = "Mixed"
            elif has_milco:
                category = "MILCO-only"
            elif has_nombo:
                category = "NOMBO-only"
            else:
                category = "Empty"

            year_match = re.search(
                r"(2010|2015|2017|2018|2021|20\d{2})",
                image_path.name,
            )
            year = year_match.group(1) if year_match else "Unknown"

            rows.append(
                {
                    "split": split,
                    "image_name": image_path.name,
                    "year": year,
                    "category": category,
                    "n_objects": len(classes),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "presentation_dataset_audit.csv", index=False)
    print(f"wrote {TABLES / 'presentation_dataset_audit.csv'}")
    return df


def make_dataset_audit_plots() -> None:
    df = collect_dataset_audit_rows()
    if df.empty:
        print("Dataset audit skipped: no rows found.")
        return

    overall = df["category"].value_counts().reindex(
        CATEGORY_ORDER,
        fill_value=0,
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        overall.index,
        overall.values,
        color=[CATEGORY_COLORS[c] for c in overall.index],
    )
    plt.ylabel("Number of images")
    plt.xlabel("Image category")
    plt.title("Overall image category distribution")
    plt.grid(axis="y", alpha=0.25)

    for i, value in enumerate(overall.values):
        plt.text(i, value, str(int(value)), ha="center", va="bottom")

    savefig("01_overall_image_category_distribution.png")

    by_split = pd.crosstab(df["split"], df["category"])
    by_split = by_split.reindex(
        index=SPLIT_ORDER,
        columns=CATEGORY_ORDER,
        fill_value=0,
    )

    by_split.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5),
        color=[CATEGORY_COLORS[c] for c in CATEGORY_ORDER],
    )
    plt.ylabel("Number of images")
    plt.xlabel("Split")
    plt.title("Image category distribution by split")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    savefig("02_image_category_distribution_by_split.png")

    by_year = pd.crosstab(df["year"], df["category"])
    by_year = by_year.reindex(columns=CATEGORY_ORDER, fill_value=0)

    plt.figure(figsize=(9, 5))
    bottom = np.zeros(len(by_year.index))
    for category in CATEGORY_ORDER:
        values = by_year[category].to_numpy()
        plt.bar(
            by_year.index,
            values,
            bottom=bottom,
            label=category,
            color=CATEGORY_COLORS[category],
        )
        bottom += values

    plt.ylabel("Number of images")
    plt.xlabel("Acquisition year")
    plt.title("Image category distribution by acquisition year")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(title="Category")
    savefig("03_image_category_distribution_by_year.png")


def make_model_comparison_plots() -> None:
    summary = summarize_final_runs()

    plot_bar_with_sem(
        summary,
        mean_col="test_map50_mean",
        sem_col="test_map50_sem",
        ylabel="Test mAP50",
        title="Held-out test mAP50 across final settings",
        filename="04_test_map50_across_final_settings.png",
    )

    plot_bar_with_sem(
        summary,
        mean_col="test_map50_95_mean",
        sem_col="test_map50_95_sem",
        ylabel="Test mAP50-95",
        title="Held-out test mAP50-95 across final settings",
        filename="05_test_map50_95_across_final_settings.png",
    )

    # Efficiency plots compare baseline with selected model.
    efficiency = summary[
        summary["setting"].isin(
            ["YOLOv4 default", "YOLO26n YOLOv4-style aug"]
        )
    ].copy()

    efficiency["runtime_minutes_mean"] = (
        efficiency["runtime_seconds_mean"] / 60
    )
    efficiency["runtime_minutes_sem"] = efficiency["runtime_seconds_sem"] / 60

    plot_bar_with_sem(
        efficiency,
        mean_col="runtime_minutes_mean",
        sem_col="runtime_minutes_sem",
        ylabel="Training runtime (minutes)",
        title="Training runtime: YOLOv4 baseline vs selected YOLO26n",
        filename="06_runtime_yolov4_vs_selected_yolo26n.png",
    )

    plot_bar_with_sem(
        efficiency,
        mean_col="best_weights_mb_mean",
        sem_col="best_weights_mb_sem",
        ylabel="Best weights file size (MB)",
        title="Model size: YOLOv4 baseline vs selected YOLO26n",
        filename="07_model_size_yolov4_vs_selected_yolo26n.png",
    )


def summarize_class_level_ap50() -> pd.DataFrame:
    df = add_setting(read_csv(TABLES / "final_class_results.csv"))

    class_col = find_col(df, ["class_name", "class"], required=False)
    ap50_col = find_col(df, ["ap50", "class_ap50", "map50"], required=False)

    if class_col is None or ap50_col is None:
        raise KeyError("Could not find class_name/class and AP50 columns.")

    rows = []
    for setting in SETTING_ORDER:
        setting_df = df[df["setting"] == setting]
        if setting_df.empty:
            continue

        for class_name in ["MILCO", "NOMBO"]:
            class_values = setting_df[class_col].astype(str).str.upper()
            class_mask = class_values == class_name
            class_df = setting_df[class_mask]
            if class_df.empty:
                continue

            ap50_mean, ap50_sem = mean_sem(class_df[ap50_col])
            rows.append(
                {
                    "setting": setting,
                    "class_name": class_name,
                    "ap50_mean": ap50_mean,
                    "ap50_sem": ap50_sem,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        TABLES / "presentation_class_level_ap50_summary.csv",
        index=False,
    )
    print(f"wrote {TABLES / 'presentation_class_level_ap50_summary.csv'}")
    return summary


def fmt_mean_sem(mean_value: float, sem_value: float, digits: int = 3) -> str:
    if pd.isna(mean_value):
        return ""
    if pd.isna(sem_value):
        sem_value = 0.0
    pm = chr(177)
    return f"{mean_value:.{digits}f} {pm} {sem_value:.{digits}f}"


def make_all_metrics_overview_table() -> None:
    summary = summarize_final_runs().copy()
    summary["setting"] = pd.Categorical(
        summary["setting"],
        categories=SETTING_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("setting")

    rows = []

    for _, row in summary.iterrows():
        setting = str(row["setting"])

        out = {
            "Model": setting,
        }

        metric_specs = [
            ("test_precision", "Test precision", 3),
            ("test_recall", "Test recall", 3),
            ("test_f1", "Test F1", 3),
            ("test_map50", "Test mAP50", 3),
            ("test_map50_95", "Test mAP50-95", 3),
        ]

        for base, label, digits in metric_specs:
            mean_col = f"{base}_mean"
            sem_col = f"{base}_sem"
            if mean_col in summary.columns and sem_col in summary.columns:
                out[label] = fmt_mean_sem(row[mean_col], row[sem_col], digits)

        rows.append(out)

    table_df = pd.DataFrame(rows)
    table_df.to_csv(
        TABLES / "presentation_heldout_test_metrics_table_mean_sem.csv",
        index=False,
    )
    print(
        f"wrote "
        f"{TABLES / 'presentation_heldout_test_metrics_table_mean_sem.csv'}"
    )

    fig, ax = plt.subplots(figsize=(15, 3.3))
    ax.axis("off")
    ax.set_title(
        f"Held-out test performance: mean {chr(177)} SEM over 3 final runs",
        fontweight="bold",
        pad=14,
    )

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    savefig("04a_heldout_test_metrics_table_mean_sem.png")


def make_class_level_plot() -> None:
    summary = summarize_class_level_ap50()

    settings = [s for s in SETTING_ORDER if s in set(summary["setting"])]
    x = np.arange(len(settings))
    width = 0.35

    milco = summary[summary["class_name"] == "MILCO"]
    milco = milco.set_index("setting").reindex(settings)
    nombo = summary[summary["class_name"] == "NOMBO"]
    nombo = nombo.set_index("setting").reindex(settings)

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2,
        milco["ap50_mean"],
        width,
        yerr=milco["ap50_sem"].fillna(0),
        capsize=5,
        color="#d62728",
        label="MILCO",
    )
    plt.bar(
        x + width / 2,
        nombo["ap50_mean"],
        width,
        yerr=nombo["ap50_sem"].fillna(0),
        capsize=5,
        color="#1f77b4",
        label="NOMBO",
    )

    plt.xticks(x, settings, rotation=15, ha="right")
    plt.ylabel("Class-level AP50")
    plt.title("Class-level AP50 across final settings")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(title="Class")
    savefig("08_class_level_ap50_across_final_settings.png")


def aggregate_yolo26n_curve(file_pattern: str) -> pd.DataFrame:
    files = sorted(YOLO26N_RESULTS.glob(file_pattern))
    if not files:
        return pd.DataFrame()

    wanted_cols = [
        "epoch",
        "train/box_loss",
        "val/box_loss",
        "train/cls_loss",
        "val/cls_loss",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    frames = []
    for file in files:
        df = read_csv(file)
        available = [col for col in wanted_cols if col in df.columns]
        if "epoch" not in available:
            df["epoch"] = np.arange(len(df))
            available = ["epoch"] + [
                col for col in wanted_cols if col in df.columns
            ]

        temp = df[available].copy()
        temp["source"] = file.name
        frames.append(temp)

    combined = pd.concat(frames, ignore_index=True)

    rows = []
    value_cols = [c for c in combined.columns if c not in ["epoch", "source"]]

    for epoch, group in combined.groupby("epoch"):
        row = {"epoch": epoch}
        for col in value_cols:
            row[f"{col}_mean"], row[f"{col}_sem"] = mean_sem(group[col])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("epoch")


def plot_curve_with_band(
    df: pd.DataFrame,
    mean_col: str,
    sem_col: str,
    color: str,
    label: str,
) -> None:
    x = df["epoch"].to_numpy()
    y = df[mean_col].to_numpy()
    e = df[sem_col].fillna(0).to_numpy()

    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y - e, y + e, color=color, alpha=0.15)


def make_yolo26n_plots() -> None:
    final_selected = aggregate_yolo26n_curve("*yolov4_style*_results.csv")
    no_aug = aggregate_yolo26n_curve("*no_aug*_results.csv")
    with_aug = aggregate_yolo26n_curve("*yolov4_style*_results.csv")

    if not final_selected.empty:
        final_selected.to_csv(
            TABLES / "presentation_yolo26n_selected_curves.csv",
            index=False,
        )
        print(f"wrote {TABLES / 'presentation_yolo26n_selected_curves.csv'}")

        plt.figure(figsize=(9, 5))
        plot_curve_with_band(
            final_selected,
            "train/box_loss_mean",
            "train/box_loss_sem",
            "#1f77b4",
            "Train",
        )
        plot_curve_with_band(
            final_selected,
            "val/box_loss_mean",
            "val/box_loss_sem",
            "#d62728",
            "Validation",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Box loss")
        plt.title("Selected YOLO26n model: train vs validation box loss")
        plt.grid(alpha=0.25)
        plt.legend(title="Split")
        savefig("09_selected_yolo26n_train_vs_val_box_loss.png")

        plt.figure(figsize=(9, 5))
        plot_curve_with_band(
            final_selected,
            "train/cls_loss_mean",
            "train/cls_loss_sem",
            "#1f77b4",
            "Train",
        )
        plot_curve_with_band(
            final_selected,
            "val/cls_loss_mean",
            "val/cls_loss_sem",
            "#d62728",
            "Validation",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Classification loss")
        plt.title(
            "Selected YOLO26n model: train vs validation "
            "classification loss"
        )
        plt.grid(alpha=0.25)
        plt.legend(title="Split")
        savefig("10_selected_yolo26n_train_vs_val_cls_loss.png")

        plt.figure(figsize=(9, 5))
        plot_curve_with_band(
            final_selected,
            "metrics/mAP50(B)_mean",
            "metrics/mAP50(B)_sem",
            "#1f77b4",
            "mAP50",
        )
        plot_curve_with_band(
            final_selected,
            "metrics/mAP50-95(B)_mean",
            "metrics/mAP50-95(B)_sem",
            "#d62728",
            "mAP50-95",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Validation score")
        plt.ylim(0, 1)
        plt.title("Selected YOLO26n model: validation mAP curves")
        plt.grid(alpha=0.25)
        plt.legend()
        savefig("11_selected_yolo26n_validation_map_curves.png")

    if not no_aug.empty and not with_aug.empty:
        no_aug.to_csv(
            TABLES / "presentation_yolo26n_no_aug_curves.csv",
            index=False,
        )
        with_aug.to_csv(
            TABLES / "presentation_yolo26n_with_aug_curves.csv",
            index=False,
        )
        print(f"wrote {TABLES / 'presentation_yolo26n_no_aug_curves.csv'}")
        print(f"wrote {TABLES / 'presentation_yolo26n_with_aug_curves.csv'}")

        plt.figure(figsize=(9, 5))
        plot_curve_with_band(
            no_aug,
            "val/box_loss_mean",
            "val/box_loss_sem",
            "#9ecae1",
            "No augmentation",
        )
        plot_curve_with_band(
            with_aug,
            "val/box_loss_mean",
            "val/box_loss_sem",
            "#1f77b4",
            "YOLOv4-style augmentation",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Validation box loss")
        plt.title("YOLO26n: validation box loss with vs without augmentation")
        plt.grid(alpha=0.25)
        plt.legend(title="Setting")
        savefig("12_yolo26n_val_box_loss_aug_vs_noaug.png")

        plt.figure(figsize=(9, 5))
        plot_curve_with_band(
            no_aug,
            "metrics/mAP50-95(B)_mean",
            "metrics/mAP50-95(B)_sem",
            "#9ecae1",
            "No augmentation",
        )
        plot_curve_with_band(
            with_aug,
            "metrics/mAP50-95(B)_mean",
            "metrics/mAP50-95(B)_sem",
            "#1f77b4",
            "YOLOv4-style augmentation",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Validation mAP50-95")
        plt.ylim(0, 1)
        plt.title("YOLO26n: validation mAP50-95 with vs without augmentation")
        plt.grid(alpha=0.25)
        plt.legend(title="Setting")
        savefig("13_yolo26n_val_map5095_aug_vs_noaug.png")


def load_yolov4_training_plan() -> pd.DataFrame:
    df = read_csv(YOLOV4_PLAN)

    job_col = find_col(df, ["job_index", "index"], required=False)
    model_col = find_col(df, ["model"], required=False)
    aug_col = find_col(df, ["augmentation", "aug"], required=False)

    if job_col is None or model_col is None or aug_col is None:
        raise KeyError(
            "final_training_plan.csv must contain job_index, model "
            "and augmentation."
        )

    df = df[[job_col, model_col, aug_col]].copy()
    df.columns = ["job_index", "model", "augmentation"]
    df = add_setting(df)
    return df[["job_index", "setting"]]


def parse_yolov4_log(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    loss_rows = []
    map_rows = []

    current_iter = None
    text = log_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    loss_pattern = re.compile(
        r"^\s*(\d+):.*?,\s*([0-9.]+)\s+avg loss",
        re.IGNORECASE,
    )
    map_pattern = re.compile(
        r"mAP@0\.50.*?=\s*([0-9.]+)",
        re.IGNORECASE,
    )
    alt_map_pattern = re.compile(
        r"mean average precision.*?=\s*([0-9.]+)",
        re.IGNORECASE,
    )

    for line in text:
        loss_match = loss_pattern.search(line)
        if loss_match:
            current_iter = int(loss_match.group(1))
            avg_loss = float(loss_match.group(2))
            loss_rows.append(
                {
                    "iteration": current_iter,
                    "avg_loss": avg_loss,
                }
            )
            continue

        map_match = map_pattern.search(line) or alt_map_pattern.search(line)
        if map_match and current_iter is not None:
            map_value = float(map_match.group(1))
            if map_value > 1:
                map_value /= 100.0

            map_rows.append(
                {
                    "iteration": current_iter,
                    "map50": map_value,
                }
            )

    return pd.DataFrame(loss_rows), pd.DataFrame(map_rows)


def aggregate_yolov4_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    plan = load_yolov4_training_plan()

    loss_frames = []
    map_frames = []

    for log_path in sorted(YOLOV4_LOGS.glob("final_*.out")):
        match = re.search(r"_(\d+)\.out$", log_path.name)
        if not match:
            continue

        job_index = match.group(1)
        plan_row = plan[plan["job_index"].astype(str) == str(job_index)]
        if plan_row.empty:
            continue

        setting = plan_row.iloc[0]["setting"]
        loss_df, map_df = parse_yolov4_log(log_path)

        if not loss_df.empty:
            loss_df = loss_df.copy()
            loss_df["setting"] = setting
            loss_df["source"] = log_path.name
            loss_frames.append(loss_df)

        if not map_df.empty:
            map_df = map_df.copy()
            map_df["setting"] = setting
            map_df["source"] = log_path.name
            map_frames.append(map_df)

    all_loss = (
        pd.concat(loss_frames, ignore_index=True)
        if loss_frames
        else pd.DataFrame()
    )
    all_map = (
        pd.concat(map_frames, ignore_index=True)
        if map_frames
        else pd.DataFrame()
    )

    loss_summary_rows = []
    if not all_loss.empty:
        grouped_loss = all_loss.groupby(["setting", "iteration"])
        for (setting, iteration), group in grouped_loss:
            loss_mean, loss_sem = mean_sem(group["avg_loss"])
            loss_summary_rows.append(
                {
                    "setting": setting,
                    "iteration": iteration,
                    "avg_loss_mean": loss_mean,
                    "avg_loss_sem": loss_sem,
                }
            )

    map_summary_rows = []
    if not all_map.empty:
        grouped_map = all_map.groupby(["setting", "iteration"])
        for (setting, iteration), group in grouped_map:
            map_mean, map_sem = mean_sem(group["map50"])
            map_summary_rows.append(
                {
                    "setting": setting,
                    "iteration": iteration,
                    "map50_mean": map_mean,
                    "map50_sem": map_sem,
                }
            )

    loss_summary = pd.DataFrame(loss_summary_rows).sort_values(
        ["setting", "iteration"]
    )
    map_summary = pd.DataFrame(map_summary_rows).sort_values(
        ["setting", "iteration"]
    )

    loss_summary.to_csv(
        TABLES / "presentation_yolov4_loss_curves.csv",
        index=False,
    )
    map_summary.to_csv(
        TABLES / "presentation_yolov4_map_curves.csv",
        index=False,
    )
    print(f"wrote {TABLES / 'presentation_yolov4_loss_curves.csv'}")
    print(f"wrote {TABLES / 'presentation_yolov4_map_curves.csv'}")

    return loss_summary, map_summary


def make_yolov4_plots() -> None:
    loss_summary, map_summary = aggregate_yolov4_curves()

    if not loss_summary.empty:
        plt.figure(figsize=(9, 5))
        visible_loss_values = []

        for setting in ["YOLOv4 default", "YOLOv4 no aug"]:
            part = loss_summary[loss_summary["setting"] == setting].copy()
            part = part.sort_values("iteration")
            part = part[part["iteration"] >= 200]

            if part.empty:
                continue

            x = part["iteration"].to_numpy()
            y = part["avg_loss_mean"].to_numpy()
            e = part["avg_loss_sem"].fillna(0).to_numpy()
            color = SETTING_COLORS[setting]

            visible_loss_values.extend(y.tolist())

            plt.plot(x, y, color=color, label=setting)
            plt.fill_between(x, y - e, y + e, color=color, alpha=0.15)

        plt.xlabel("Iteration")
        plt.ylabel("Training average loss")
        plt.title("YOLOv4: training average loss after warm-up")
        plt.grid(alpha=0.25)
        plt.legend(title="Setting")

        if visible_loss_values:
            upper = np.nanpercentile(visible_loss_values, 98) * 1.15
            if np.isfinite(upper) and upper > 0:
                plt.ylim(0, upper)

        savefig("14_yolov4_training_avg_loss_after_warmup.png")

    if not map_summary.empty:
        plt.figure(figsize=(9, 5))

        for setting in ["YOLOv4 default", "YOLOv4 no aug"]:
            part = map_summary[map_summary["setting"] == setting].copy()
            part = part.sort_values("iteration")

            if part.empty:
                continue

            x = part["iteration"].to_numpy()
            y = part["map50_mean"].to_numpy()
            e = part["map50_sem"].fillna(0).to_numpy()
            color = SETTING_COLORS[setting]

            plt.plot(x, y, color=color, label=setting)
            plt.fill_between(x, y - e, y + e, color=color, alpha=0.15)

        plt.xlabel("Iteration")
        plt.ylabel("Validation mAP50")
        plt.ylim(0, 1)
        plt.title("YOLOv4: validation mAP50 with vs without augmentation")
        plt.grid(alpha=0.25)
        plt.legend(title="Setting")
        savefig("15_yolov4_validation_map50_aug_vs_noaug.png")


def main() -> None:
    make_dataset_audit_plots()
    make_model_comparison_plots()
    make_all_metrics_overview_table()
    make_class_level_plot()
    make_yolo26n_plots()
    make_yolov4_plots()

    print("\nDone.")
    print(f"Figures: {FIGS}")
    print(f"Tables:  {TABLES}")


if __name__ == "__main__":
    main()
