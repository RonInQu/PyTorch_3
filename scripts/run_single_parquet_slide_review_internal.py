from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_slide_style_similarity_views_internal import plot_file_slide_style_from_df


NOISE_VALUE = 5.0
SAMPLE_RATE = 150
DEFAULT_MIN_TISSUE_DURATION_SEC = 7.0

blood_events = [6, 12]
clot_events = [7, 11]
wall_events = [23]
contrast_events = [8]
saline_events = [15]
artifact_events = contrast_events + saline_events
tissue_events = blood_events + clot_events + wall_events
graphics_events = tissue_events + artifact_events

event_colors = {
    6: ("black", "blood"),
    12: ("black", "blood"),
    7: ("red", "clot"),
    11: ("red", "clot"),
    23: ("blue", "wall"),
    8: ("magenta", "contrast"),
    15: ("cyan", "saline"),
}


def crop_to_blood_range(df: pd.DataFrame, event_col: str, time_col: str) -> tuple[pd.DataFrame, tuple[float | None, float | None]]:
    blood_mask = df[event_col].isin(blood_events)
    if not blood_mask.any():
        return df.copy(), (df[time_col].min(), df[time_col].max())

    first_blood_idx = blood_mask.idxmax()
    last_blood_idx = blood_mask[::-1].idxmax()
    cropped = df.loc[first_blood_idx:last_blood_idx].copy()
    first_time = float(df.loc[first_blood_idx, time_col])
    last_time = float(df.loc[last_blood_idx, time_col])
    return cropped, (first_time, last_time)


def map_da_label(led_state) -> int:
    if pd.isna(led_state) or led_state in [0, "", None]:
        return 0
    try:
        led = int(float(led_state))
    except (ValueError, TypeError):
        return 0
    if led == 2:
        return 0
    if led in (4, 5):
        return 1
    if led == 7:
        return 2
    return 0


def assign_numeric_label(event: int) -> int:
    if event in blood_events:
        return 0
    if event in clot_events:
        return 1
    if event in wall_events:
        return 2
    return 0


def blank_short_tissue_events(
    df: pd.DataFrame,
    resistance_col: str,
    label_col: str,
    blood_median: float,
    min_dur_sec: float | None,
) -> tuple[int, int]:
    if not min_dur_sec or min_dur_sec <= 0:
        return 0, 0

    min_samples = int(min_dur_sec * SAMPLE_RATE)
    labels = df[label_col].to_numpy()
    n_events_blanked = 0
    n_samples_blanked = 0

    for cls in [1, 2]:
        cls_mask = (labels == cls).astype(int)
        diff = np.diff(np.concatenate([[0], cls_mask, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for start, end in zip(starts, ends):
            if (end - start) < min_samples:
                idx = df.index[start:end]
                noise = NOISE_VALUE * np.random.randn(end - start)
                df.loc[idx, resistance_col] = blood_median + noise
                df.loc[idx, label_col] = 0
                n_events_blanked += 1
                n_samples_blanked += end - start

    return n_events_blanked, n_samples_blanked


def build_testing_output(df_cropped: pd.DataFrame, time_col: str, resistance_col: str) -> pd.DataFrame:
    df_testing = df_cropped.copy()
    if "curr_led_state" in df_testing.columns:
        df_testing["da_label"] = df_testing["curr_led_state"].apply(map_da_label)
    else:
        df_testing["da_label"] = 0

    df_testing[resistance_col] = df_testing[resistance_col].clip(upper=5000)
    df_testing["label"] = df_testing["event_type_1"].apply(assign_numeric_label)

    return pd.DataFrame(
        {
            "timeInMS": (df_testing[time_col] * 1000).astype(int),
            "magRLoadAdjusted": df_testing[resistance_col],
            "label": df_testing["label"],
            "da_label": df_testing["da_label"],
        }
    ).reset_index(drop=True)


def build_training_view(df_cropped: pd.DataFrame, time_col: str, resistance_col: str, min_tissue_duration_sec: float | None) -> pd.DataFrame:
    df_training = df_cropped.copy()
    if "curr_led_state" in df_training.columns:
        df_training["da_label"] = df_training["curr_led_state"].apply(map_da_label)
    else:
        df_training["da_label"] = 0

    df_training["label"] = df_training["event_type_1"].apply(assign_numeric_label)

    tissue_mask = df_training["event_type_1"].isin(tissue_events)
    outlier_mask = df_training[resistance_col] > 5000
    blank_mask = (~tissue_mask) | outlier_mask

    if blank_mask.any():
        blood_median = df_training.loc[df_training["event_type_1"].isin(blood_events), resistance_col].median()
        if pd.isna(blood_median):
            blood_median = df_training[resistance_col].median()
        noise = NOISE_VALUE * np.random.randn(int(blank_mask.sum()))
        df_training.loc[blank_mask, resistance_col] = blood_median + noise
        df_training.loc[blank_mask, "label"] = 0

    if min_tissue_duration_sec:
        blood_median_for_short = df_training.loc[df_training["label"] == 0, resistance_col].median()
        if pd.isna(blood_median_for_short):
            blood_median_for_short = df_training[resistance_col].median()
        blank_short_tissue_events(
            df_training,
            resistance_col,
            "label",
            float(blood_median_for_short),
            min_tissue_duration_sec,
        )

    return pd.DataFrame(
        {
            "timeInMS": (df_training[time_col] * 1000).astype(int),
            "magRLoadAdjusted": df_training[resistance_col],
            "label": df_training["label"],
            "da_label": df_training["da_label"],
        }
    ).reset_index(drop=True)


def save_graphics(
    study_name: str,
    df_cropped: pd.DataFrame,
    df_training_view: pd.DataFrame,
    time_col: str,
    resistance_col: str,
    graphics_dir: Path,
    min_tissue_duration_sec: float | None,
) -> None:
    df_graphics = df_cropped.copy()
    if "curr_led_state" in df_graphics.columns:
        df_graphics["da_label"] = df_graphics["curr_led_state"].apply(map_da_label)
    else:
        df_graphics["da_label"] = 0

    gfx_highlighted_mask = df_graphics["event_type_1"].isin(graphics_events)
    default_blank_mask = (df_graphics["da_label"] == 0) & (~gfx_highlighted_mask)
    outlier_mask = (df_graphics[resistance_col] > 5000) & (~df_graphics["event_type_1"].isin(artifact_events))
    blank_mask = default_blank_mask | outlier_mask

    if blank_mask.any():
        blood_median = df_graphics.loc[df_graphics["event_type_1"].isin(blood_events), resistance_col].median()
        if pd.isna(blood_median):
            blood_median = df_graphics[resistance_col].median()
        noise = NOISE_VALUE * np.random.randn(int(blank_mask.sum()))
        df_graphics.loc[blank_mask, resistance_col] = blood_median + noise

    plt.figure(figsize=(14, 7))
    plt.plot(df_graphics[time_col], df_graphics[resistance_col], "k-", lw=0.8, alpha=0.7, zorder=1)

    seen_labels: set[str] = set()
    for ev_type, (color, label_name) in event_colors.items():
        mask = df_graphics["event_type_1"] == ev_type
        if mask.any():
            plt.scatter(
                df_graphics.loc[mask, time_col],
                df_graphics.loc[mask, resistance_col],
                color=color,
                s=6,
                label=label_name if label_name not in seen_labels else None,
                zorder=5,
            )
            seen_labels.add(label_name)

    plt.title(
        f"{study_name} - First to Last Blood (5-class view)\n"
        "magRLoadAdjusted = raw - baseline + 800 | Artifacts shown, not blanked"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Resistance (ohms)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.savefig(graphics_dir / f"{study_name}_first_last_blood.png", dpi=250, bbox_inches="tight")
    plt.close()

    train_label_colors = {0: ("black", "blood"), 1: ("red", "clot"), 2: ("blue", "wall")}
    plt.figure(figsize=(14, 7))
    time_sec = df_training_view["timeInMS"].to_numpy(dtype=np.float64) / 1000.0
    r_vals = df_training_view["magRLoadAdjusted"].to_numpy(dtype=np.float32)
    labels = df_training_view["label"].to_numpy(dtype=np.int16)

    plt.plot(time_sec, r_vals, "k-", lw=0.8, alpha=0.3, zorder=1)
    seen_labels.clear()
    for lbl, (color, label_name) in train_label_colors.items():
        mask = labels == lbl
        if mask.any():
            plt.scatter(
                time_sec[mask],
                r_vals[mask],
                color=color,
                s=6,
                label=label_name if label_name not in seen_labels else None,
                zorder=5,
            )
            seen_labels.add(label_name)

    plt.title(
        f"{study_name} - Training View (as seen by model)\n"
        f"After blanking: artifacts, outliers, short events (<{min_tissue_duration_sec}s) -> blood"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Resistance (ohms)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.savefig(graphics_dir / f"{study_name}_training_view.png", dpi=250, bbox_inches="tight")
    plt.close()


def process_one_file(input_parquet: Path, output_root: Path, min_tissue_duration_sec: float | None) -> dict[str, Path]:
    study_name = input_parquet.stem
    graphics_dir = output_root / "graphics"
    slide_dir = output_root / "slide_style_similarity"

    for folder in [graphics_dir, slide_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    df1 = pd.read_parquet(input_parquet)
    event_col = "event_type_1"
    time_col = df1.columns[0]
    imp_col_name = "imp" if "imp" in df1.columns else df1.columns[2]
    baseline_col_name = "blood_baseline" if "blood_baseline" in df1.columns else df1.columns[3]

    df1["magRLoadAdjusted"] = df1[imp_col_name] - df1[baseline_col_name] + 800
    resistance_col = "magRLoadAdjusted"
    df1[resistance_col] = df1[resistance_col].astype("float64")

    df_cropped, _ = crop_to_blood_range(df1, event_col, time_col)
    df_training_view = build_training_view(df_cropped, time_col, resistance_col, min_tissue_duration_sec)
    df_testing_out = build_testing_output(df_cropped, time_col, resistance_col)

    save_graphics(
        study_name=study_name,
        df_cropped=df_cropped,
        df_training_view=df_training_view,
        time_col=time_col,
        resistance_col=resistance_col,
        graphics_dir=graphics_dir,
        min_tissue_duration_sec=min_tissue_duration_sec,
    )

    slide_png = slide_dir / f"{study_name}_slide_style.png"
    plot_file_slide_style_from_df(study_name, df_testing_out, slide_png)

    return {
        "graphics_first_last_blood": graphics_dir / f"{study_name}_first_last_blood.png",
        "graphics_training_view": graphics_dir / f"{study_name}_training_view.png",
        "slide_png": slide_png,
        "slide_html": slide_png.with_suffix(".html"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single raw parquet through internal review output generation. Outputs only graphics and slide_style_similarity folders."
    )
    parser.add_argument("--input-parquet", type=Path, required=True, help="Path to a full raw parquet file.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output folder root. Defaults to <input parent>/processedResults.",
    )
    parser.add_argument(
        "--min-tissue-duration-sec",
        type=float,
        default=DEFAULT_MIN_TISSUE_DURATION_SEC,
        help="Minimum clot/wall duration used by the training-view graphics branch.",
    )
    args = parser.parse_args()

    input_parquet = args.input_parquet.resolve()
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")

    output_root = args.output_root.resolve() if args.output_root else input_parquet.parent / "processedResults"
    outputs = process_one_file(input_parquet, output_root, args.min_tissue_duration_sec)

    print("Saved outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()