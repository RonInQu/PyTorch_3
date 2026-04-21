#!/usr/bin/env python3
"""
Batch HF-Noise-Removed V93 Comparison Script

Removes high-frequency noise from the original ~154 Hz impedance signal while
retaining the cardiac pulse.  This is the complement of the CardiacPulseRemoved
case (which removes cardiac but keeps HF noise).

A Butterworth low-pass filter (cutoff ~10 Hz) is applied to the DB rec_df
impedance within the labeled segment.  The cardiac pulse (~1-2 Hz) passes
through; higher-frequency noise is attenuated.

Flow:
  1. Load full rec_df from database (original impedance at ~154 Hz)
  2. Apply low-pass filter to imp within the labeled segment time range
     (cardiac retained, HF noise removed)
  3. Run v93 sim on the modified rec_df (full recording context preserved)
  4. Classify v93 results against human annotations
  5. Also classify ML predictions (da_label from CardiacPulseRemoved parquet)
  6. Save results to ./output_hfnoise_removed/

Usage:
    python batch_hfnoise_removed_v93_compare.py
    python batch_hfnoise_removed_v93_compare.py --no-plots
    python batch_hfnoise_removed_v93_compare.py --force
    python batch_hfnoise_removed_v93_compare.py --cutoff 15
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse
import io
import contextlib
from pathlib import Path
from datetime import datetime
import traceback
from scipy.signal import butter, sosfiltfilt

# Import IDP modules
from idp.schema import *
from idp.schema import code_by_light_pattern
from idp.helpers import *
from idp import database as db
from idp import plots
from idp import classify_tools
from idp import sim


PARQUET_DIR = Path("./CardiacPulseRemoved")
OUTPUT_ROOT = Path("./output_hfnoise_removed")

# Default low-pass cutoff frequency in Hz.
# Cardiac pulse is ~1-2 Hz; we want to pass it through while removing HF noise.
DEFAULT_CUTOFF_HZ = 10.0
FILTER_ORDER = 4


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_ROOT.mkdir(exist_ok=True)

    confusion_matrices_dir = OUTPUT_ROOT / "confusion_matrices"
    plots_dir = OUTPUT_ROOT / "plots"
    logs_dir = OUTPUT_ROOT / "logs"

    confusion_matrices_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    return confusion_matrices_dir, plots_dir, logs_dir


def extract_expt_id_from_filename(filename):
    """Extract experiment ID from parquet filename.

    Expected format:  {EXPT_ID}_labeled_segment_denoised_150Hz.parquet
    """
    basename = os.path.basename(filename)
    suffix = "_labeled_segment_denoised_150Hz.parquet"
    if basename.endswith(suffix):
        return basename[:-len(suffix)]
    return None


def load_experiment_data(expt_id, expt_df, event_df, lgtevent_df):
    """Load all experiment data from the database.

    Returns the full rec_df (needed so the sim sees the whole recording from
    the start for proper calibration), plus light events and classifier markers.
    """
    try:
        expt_light_df = lgtevent_df[lgtevent_df.expt_id == expt_id].copy()
        expt_rec_df = db.expt_ids_to_rec_df(expt_id).copy()
        expt_event_df = event_df_split_by_expt_id(event_df)[expt_id]

        start_event = expt_event_df[expt_event_df.event_type == EventType.START_HERE_FOR_CLASSIFIER]
        end_event = expt_event_df[expt_event_df.event_type == EventType.END_HERE_FOR_CLASSIFIER]

        if len(start_event) == 0 or len(end_event) == 0:
            print(f"  ⚠ No START/END_HERE_FOR_CLASSIFIER markers found for {expt_id}")
            return None

        start_time_sec = start_event.iloc[0].start
        stop_time_sec = end_event.iloc[0].stop

        return {
            "expt_light_df": expt_light_df,
            "expt_rec_df": expt_rec_df,
            "expt_event_df": expt_event_df,
            "start_time_sec": start_time_sec,
            "stop_time_sec": stop_time_sec,
        }
    except Exception as e:
        print(f"  ✗ Error loading experiment data for {expt_id}: {e}")
        return None


ML_TO_LED_STATE = {
    0: LedState.IN_BODY.value,   # 2
    1: LedState.CLOT.value,      # 4
    2: LedState.LATCH.value,     # 7
}


def apply_lowpass_filter(expt_rec_df, parquet_path, cutoff_hz):
    """Apply a low-pass Butterworth filter to remove HF noise while keeping cardiac.

    Filters the imp column within the time range covered by the parquet file.
    Samples outside that range retain their original imp values (sim warmup).

    Returns: (modified rec_df, n_filtered, n_total)
    """
    try:
        # Determine the time range from the parquet (labeled segment)
        parquet_df = pd.read_parquet(parquet_path)
        parquet_tmin = parquet_df['timeInMS'].min() / 1000.0
        parquet_tmax = parquet_df['timeInMS'].max() / 1000.0

        expt_rec_df = expt_rec_df.sort_values('time_sec').reset_index(drop=True)
        expt_rec_df['time_sec'] = expt_rec_df['time_sec'].astype('float64')
        expt_rec_df['imp'] = expt_rec_df['imp'].astype('float64')

        # Select samples within the labeled segment for filtering
        mask = (expt_rec_df['time_sec'] >= parquet_tmin) & (expt_rec_df['time_sec'] <= parquet_tmax + 1.0)
        n_filtered = int(mask.sum())
        n_total = len(expt_rec_df)

        if n_filtered < 20:
            print(f"  ✗ Too few samples ({n_filtered}) in parquet time range for filtering")
            return None, 0, 0

        # Estimate sampling rate from the segment
        segment_times = expt_rec_df.loc[mask, 'time_sec'].values
        dt = np.median(np.diff(segment_times))
        fs = 1.0 / dt

        # Design Butterworth low-pass filter
        nyquist = fs / 2.0
        if cutoff_hz >= nyquist:
            print(f"  ⚠ Cutoff {cutoff_hz} Hz >= Nyquist {nyquist:.1f} Hz; using {nyquist*0.9:.1f} Hz")
            cutoff_hz = nyquist * 0.9

        sos = butter(FILTER_ORDER, cutoff_hz / nyquist, btype='low', output='sos')

        # Apply zero-phase filter to the segment
        segment_imp = expt_rec_df.loc[mask, 'imp'].values.copy()
        filtered_imp = sosfiltfilt(sos, segment_imp)

        expt_rec_df.loc[mask, 'imp'] = filtered_imp

        return expt_rec_df, n_filtered, n_total

    except Exception as e:
        print(f"  ✗ Error applying low-pass filter: {e}")
        traceback.print_exc()
        return None, 0, 0


def load_ml_predictions(expt_id, expt_rec_df, parquet_path):
    """Load ML predictions (da_label) from the 150 Hz parquet and merge onto rec_df."""
    try:
        parquet_df = pd.read_parquet(parquet_path)

        if 'da_label' not in parquet_df.columns:
            print(f"  ✗ No 'da_label' column in {parquet_path}")
            return None

        expt_rec_df = expt_rec_df.sort_values('time_sec').reset_index(drop=True)

        parquet_sorted = pd.DataFrame()
        parquet_sorted['time'] = (parquet_df['timeInMS'].values / 1000.0).astype('float64')
        parquet_sorted['prediction'] = parquet_df['da_label'].values
        parquet_sorted = parquet_sorted.sort_values('time').reset_index(drop=True)

        expt_rec_df['time_sec'] = expt_rec_df['time_sec'].astype('float64')

        merged = pd.merge_asof(
            expt_rec_df[['time_sec']],
            parquet_sorted[['time', 'prediction']],
            left_on='time_sec',
            right_on='time',
            direction='backward'
        )

        mask = merged['prediction'].notna()
        expt_rec_df.loc[mask, 'led_state'] = merged.loc[mask, 'prediction'].map(ML_TO_LED_STATE).astype(int)

        return expt_rec_df
    except Exception as e:
        print(f"  ✗ Error loading ML predictions for {expt_id}: {e}")
        return None


def run_simulator(expt_rec_df):
    """Run the v93 simulator on the impedance data."""
    try:
        sim_expt_rec_df = sim.sim_gen2(
            rec_df=expt_rec_df,
            base_config=None,
            config_overrides=dict(),
            show_stdout=False,
            simple_output=False,
        )
        sim_expt_rec_df["led_state"] = sim_expt_rec_df["curr_led_state"]
        return sim_expt_rec_df
    except Exception as e:
        print(f"  ✗ Error running simulator: {e}")
        return None


def label_rec_df_from_tp_df(rec_df, tp_df, prefix=""):
    """Stamp per-light-event classification results onto every sample."""
    rec_df[f"{prefix}any_error"] = 0
    rec_df[f"{prefix}cl_error"] = 0
    rec_df[f"{prefix}true_light_pattern"] = 0

    for _, light_row in tp_df.iterrows():
        span_rec_mask = (rec_df.time_sec >= light_row.start) & (rec_df.time_sec <= light_row.stop)
        rec_df.loc[span_rec_mask, f"{prefix}any_error"] = int(light_row.any_error)
        rec_df.loc[span_rec_mask, f"{prefix}cl_error"] = int(light_row.cl_error)
        rec_df.loc[span_rec_mask, f"{prefix}true_light_pattern"] = int(light_row.light_pattern.value)


def classify_and_compare(expt_id, expt_light_df, expt_rec_df, sim_expt_rec_df):
    """Classify both original and simulated LED states."""
    try:
        org_tp_df, org_imposter_df = classify_tools.abcl_classify_rec_df_per_lgtevent(
            expt_light_df, expt_rec_df
        )
        label_rec_df_from_tp_df(expt_rec_df, org_tp_df, "org_")

        sim_tp_df, sim_imposter_df = classify_tools.abcl_classify_rec_df_per_lgtevent(
            expt_light_df, sim_expt_rec_df
        )
        label_rec_df_from_tp_df(expt_rec_df, sim_tp_df, "sim_")

        return org_tp_df, org_imposter_df, sim_tp_df, sim_imposter_df
    except Exception as e:
        print(f"  ✗ Error during classification: {e}")
        return None, None, None, None


def save_confusion_matrices(expt_id, org_tp_df, org_imposter_df, sim_tp_df, sim_imposter_df, output_dir, cutoff_hz):
    """Save confusion matrices to text files."""
    try:
        output_file_txt = output_dir / f"{expt_id}_confusion_matrices.txt"

        def capture_summary(func, *args, **kwargs):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                func(*args, **kwargs)
            return buf.getvalue()

        ml_summary_text = capture_summary(classify_tools.abcl_tp_summary, org_tp_df, org_imposter_df, True)
        v93_summary_text = capture_summary(classify_tools.abcl_tp_summary, sim_tp_df, sim_imposter_df, True)

        with open(output_file_txt, 'w') as f:
            f.write(f"HF-Noise-Removed V93 Confusion Matrix Comparison: {expt_id}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"NOTE: V93 simulator ran on low-pass filtered impedance (cutoff={cutoff_hz} Hz)\n")
            f.write("      Cardiac pulse RETAINED, high-frequency noise REMOVED\n\n")

            f.write("ML MODEL SUMMARY (da_label predictions from parquet):\n")
            f.write(ml_summary_text + "\n")
            f.write("\n")

            f.write(f"V93 SIMULATOR SUMMARY (running on LP-filtered impedance, cutoff={cutoff_hz} Hz):\n")
            f.write(v93_summary_text + "\n")
            f.write("\n")

            f.write("ML MODEL CONFUSION MATRIX\n")
            f.write("-" * 80 + "\n")
            f.write("True Pattern → Predicted Pattern Distribution\n\n")

            if len(org_imposter_df) > 0:
                ml_cm = pd.crosstab(
                    org_imposter_df['true_light_i'],
                    org_imposter_df['pred_light_i'],
                    margins=True
                )
                f.write(ml_cm.to_string())
            f.write("\n\n")

            f.write(f"V93 SIMULATOR CONFUSION MATRIX (LP-FILTERED, cutoff={cutoff_hz} Hz)\n")
            f.write("-" * 80 + "\n")
            f.write("True Pattern → Predicted Pattern Distribution\n\n")

            if len(sim_imposter_df) > 0:
                v93_cm = pd.crosstab(
                    sim_imposter_df['true_light_i'],
                    sim_imposter_df['pred_light_i'],
                    margins=True
                )
                f.write(v93_cm.to_string())
            f.write("\n")

        return str(output_file_txt)
    except Exception as e:
        print(f"  ✗ Error saving confusion matrices: {e}")
        return None


def create_plot(expt_id, expt_rec_df, expt_light_df, sim_expt_rec_df, start_time_sec, stop_time_sec, output_dir):
    """Generate and save ML vs HF-noise-removed v93 comparison plot."""
    try:
        pattern_palette = {
            LightPattern.UNKNOWN.value: "black",
            LightPattern.AIR.value: "rgb(200,200,200)",
            LightPattern.IDLE.value: "gray",
            LightPattern.CLOT_OR_LATCH.value: "purple",
            LightPattern.CLOT_AND_LATCH.value: "pink",
            LightPattern.CLOT_EXCLUSIVE.value: "darkorange",
            LightPattern.LATCH_EXCLUSIVE.value: "blue",
            LightPattern.TRACKING.value: "green",
            LightPattern.LATCH.value: "cyan",
            LightPattern.CONTRAST.value: "lightgray",
            LightPattern.CLOT_ENGAGE.value: "orange",
            LightPattern.CLOT_NOMINAL.value: "coral",
            LightPattern.WALL_TOUCH.value: "cornflowerblue",
        }

        any_error_palette = {
            0: "white",
            1: "red",
        }

        states = [
            ("org_true_light_pattern", expt_rec_df, "HUMAN ANNOTATION", False, pattern_palette,
             [f"{p.name} {code_by_light_pattern.get(p, '')}" for p in LightPattern]),
            ("led_state", expt_rec_df, "ML LED STATE", True),
            ("led_state", sim_expt_rec_df, "V93 LED STATE (HF-NOISE REMOVED)", True),
            ("org_any_error", expt_rec_df, "ML GRADER ERROR", True, any_error_palette),
            ("sim_any_error", expt_rec_df, "V93 GRADER ERROR (HF-NOISE REMOVED)", True, any_error_palette),
        ]

        traces = []

        fig = plots.expt_plot(
            expt_id=expt_id,
            rec_df=expt_rec_df,
            title=f"ML vs HF-Noise-Removed V93 Comparison: {expt_id}",
            time_range=(start_time_sec, stop_time_sec),
            imp_range=(500, 10000),
            states=states,
            traces=traces,
            thresholds=[],
            show_plot=False
        )

        output_file = output_dir / f"{expt_id}_ml_hfnoise_removed_v93_comparison.html"
        fig.write_html(
            str(output_file),
            config={
                'responsive': True,
                'displaymodeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{expt_id}_ml_hfnoise_removed_v93_comparison',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
        )
        return str(output_file)
    except Exception as e:
        print(f"  ✗ Error creating plot: {e}")
        traceback.print_exc()
        return None


def process_experiment(expt_id, parquet_file, expt_df, event_df, lgtevent_df,
                       confusion_matrices_dir, plots_dir, cutoff_hz,
                       nooplot=False, force=False):
    """Process a single experiment: low-pass filter imp, run v93 sim, classify."""
    cm_path = confusion_matrices_dir / f"{expt_id}_confusion_matrices.txt"
    plot_path = plots_dir / f"{expt_id}_ml_hfnoise_removed_v93_comparison.html"
    if not force and cm_path.exists() and (nooplot or plot_path.exists()):
        print(f"\nℹ Skipping {expt_id} (already processed)")
        return True

    print(f"\n📊 Processing: {expt_id}")

    # Load full experiment data from DB (rec_df needed for sim warmup context)
    exp_data = load_experiment_data(expt_id, expt_df, event_df, lgtevent_df)
    if exp_data is None:
        return False

    expt_light_df = exp_data["expt_light_df"]
    expt_rec_df = exp_data["expt_rec_df"]
    start_time_sec = exp_data["start_time_sec"]
    stop_time_sec = exp_data["stop_time_sec"]

    # Apply low-pass filter to remove HF noise (cardiac retained)
    print(f"  → Applying low-pass filter (cutoff={cutoff_hz} Hz) to imp...")
    expt_rec_df, n_filt, n_total = apply_lowpass_filter(expt_rec_df, parquet_file, cutoff_hz)
    if expt_rec_df is None:
        return False
    print(f"      ✓ Filtered {n_filt}/{n_total} samples ({100*n_filt/n_total:.1f}%)")

    # Load ML predictions (da_label from parquet)
    print(f"  → Loading ML predictions from parquet...")
    expt_rec_df = load_ml_predictions(expt_id, expt_rec_df, parquet_file)
    if expt_rec_df is None:
        return False

    # Run v93 simulator on the LP-filtered impedance
    print(f"  → Running v93 simulator on LP-filtered impedance...")
    sim_expt_rec_df = run_simulator(expt_rec_df.copy())
    if sim_expt_rec_df is None:
        return False

    # Classify and compare
    print(f"  → Classifying LED states...")
    org_tp_df, org_imposter_df, sim_tp_df, sim_imposter_df = classify_and_compare(
        expt_id, expt_light_df, expt_rec_df, sim_expt_rec_df
    )
    if org_tp_df is None:
        return False

    # Save confusion matrices
    print(f"  → Saving confusion matrices...")
    cm_file = save_confusion_matrices(
        expt_id, org_tp_df, org_imposter_df, sim_tp_df, sim_imposter_df,
        confusion_matrices_dir, cutoff_hz
    )
    if cm_file:
        print(f"      ✓ Saved: {cm_file}")

    # Create plot
    if not nooplot and not plot_path.exists():
        print(f"  → Creating comparison plot...")
        plot_file = create_plot(
            expt_id, expt_rec_df, expt_light_df, sim_expt_rec_df,
            start_time_sec, stop_time_sec, plots_dir
        )
        if plot_file:
            print(f"      ✓ Saved: {plot_file}")
        else:
            print(f"      ⚠ Plot failed for {expt_id}")
    elif plot_path.exists():
        print(f"  → Plot already exists; skipping")

    return True


def aggregate_summaries(confusion_matrices_dir, output_dir, cutoff_hz, expt_ids=None):
    """Extract and concatenate ABCL summary tables from confusion matrix files."""
    try:
        if expt_ids is not None:
            txt_files = sorted(
                confusion_matrices_dir / f"{eid}_confusion_matrices.txt"
                for eid in expt_ids
                if (confusion_matrices_dir / f"{eid}_confusion_matrices.txt").exists()
            )
        else:
            txt_files = sorted(confusion_matrices_dir.glob("*_confusion_matrices.txt"))
        if len(txt_files) == 0:
            print("  ⚠ No confusion matrix files found to aggregate")
            return

        ml_lines = []
        v93_lines = []
        combined_lines = []

        composite_data = {
            'ml': {'Air': [], 'Bld': [], 'Clt': [], 'Lch': [], '2:1:1': []},
            'v93': {'Air': [], 'Bld': [], 'Clt': [], 'Lch': [], '2:1:1': []}
        }

        header = "ExptID  │ Type │ Label │   TP │   FN │   FP │   TN │ Prec │ Sens │ Spec │   F1 │  MCC | #Pos | #Neg | % A  | % B  | % C  | % L  "
        separator = "────────┼──────┼───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────"

        ml_lines.append(header)
        ml_lines.append(separator)
        v93_lines.append(header)
        v93_lines.append(separator)
        combined_lines.append(header)
        combined_lines.append(separator)

        for txt_file in sorted(txt_files):
            expt_id = txt_file.stem.replace("_confusion_matrices", "")

            try:
                with open(txt_file, 'r') as f:
                    content = f.read()

                ml_section = None
                v93_section = None

                if "ML MODEL SUMMARY" in content:
                    ml_start = content.find("ML MODEL SUMMARY") + len("ML MODEL SUMMARY")
                    ml_start = content.find("\n", ml_start) + 1
                    if "V93 SIMULATOR SUMMARY" in content:
                        ml_end = content.find("V93 SIMULATOR SUMMARY")
                    else:
                        ml_end = len(content)
                    ml_section = content[ml_start:ml_end].strip()

                if "V93 SIMULATOR SUMMARY" in content:
                    v93_start = content.find("V93 SIMULATOR SUMMARY") + len("V93 SIMULATOR SUMMARY")
                    v93_start = content.find("\n", v93_start) + 1
                    if "ML MODEL CONFUSION MATRIX" in content:
                        v93_end = content.find("ML MODEL CONFUSION MATRIX")
                    else:
                        v93_end = len(content)
                    v93_section = content[v93_start:v93_end].strip()

                def extract_summary_rows(section):
                    rows = []
                    if section is None:
                        return rows
                    for line in section.split('\n'):
                        line = line.strip()
                        if not line or '─' in line or 'LABEL' in line or 'Label' in line:
                            continue
                        if any(line.startswith(label) for label in ['Air', 'Bld', 'Clt', 'Lch', '2:1:1']):
                            rows.append(line)
                    return rows

                ml_rows = extract_summary_rows(ml_section)
                v93_rows = extract_summary_rows(v93_section)

                for row in ml_rows:
                    ml_lines.append(f"{expt_id} │ ML   │ {row}")
                    parts = row.split()
                    if len(parts) > 0:
                        label = parts[0]
                        if label in composite_data['ml']:
                            composite_data['ml'][label].append(row)

                for row in v93_rows:
                    v93_lines.append(f"{expt_id} │ V93f │ {row}")
                    parts = row.split()
                    if len(parts) > 0:
                        label = parts[0]
                        if label in composite_data['v93']:
                            composite_data['v93'][label].append(row)

                for row in ml_rows:
                    combined_lines.append(f"{expt_id} │ ML   │ {row}")
                for row in v93_rows:
                    combined_lines.append(f"{expt_id} │ V93f │ {row}")

                combined_lines.append("")

            except Exception as e:
                print(f"  ⚠ Error processing {txt_file}: {e}")
                continue

        # Write ML aggregate
        ml_output = output_dir / "ML_MODEL_AGGREGATED_SUMMARY.txt"
        with open(ml_output, 'w') as f:
            f.write("ML MODEL - All Experiments Summary (HF-Noise-Removed Run)\n")
            f.write("=" * 145 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write('\n'.join(ml_lines))
            f.write("\n")
        print(f"  ✓ Saved: {ml_output}")

        # Write V93-filtered aggregate
        v93_output = output_dir / "V93_HFNOISE_REMOVED_AGGREGATED_SUMMARY.txt"
        with open(v93_output, 'w') as f:
            f.write(f"V93 SIMULATOR (HF-Noise-Removed, LP cutoff={cutoff_hz} Hz) - All Experiments Summary\n")
            f.write("=" * 145 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write('\n'.join(v93_lines))
            f.write("\n")
        print(f"  ✓ Saved: {v93_output}")

        # Write combined aggregate
        combined_output = output_dir / "COMBINED_ML_V93_HFNOISE_REMOVED_SUMMARY.txt"
        with open(combined_output, 'w') as f:
            f.write(f"COMBINED ML vs V93-HF-Noise-Removed (LP cutoff={cutoff_hz} Hz) - All Experiments Summary\n")
            f.write("=" * 145 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"V93f = V93 simulator running on LP-filtered impedance (cardiac retained, HF noise removed)\n\n")
            f.write('\n'.join(combined_lines))
            f.write("\n")
        print(f"  ✓ Saved: {combined_output}")

        # Generate composite matrices
        generate_composite_matrices(composite_data, output_dir, cutoff_hz)

    except Exception as e:
        print(f"  ✗ Error aggregating summaries: {e}")
        traceback.print_exc()


def generate_composite_matrices(composite_data, output_dir, cutoff_hz):
    """Generate composite confusion matrices with aggregated metrics and F1 scores."""
    try:
        composite_output = output_dir / "COMPOSITE_CONFUSION_MATRICES.txt"

        with open(composite_output, 'w') as f:
            f.write(f"COMPOSITE CONFUSION MATRICES - HF-Noise-Removed V93 (LP cutoff={cutoff_hz} Hz)\n")
            f.write("=" * 120 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"V93 simulator ran on LP-filtered impedance (cutoff={cutoff_hz} Hz).\n")
            f.write("Cardiac pulse RETAINED, high-frequency noise REMOVED.\n")
            f.write("TP/TN/FP/FN combined across all experiments.\n\n")

            f.write("\n" + "=" * 120 + "\n")
            f.write("ML MODEL - COMPOSITE MATRIX\n")
            f.write("=" * 120 + "\n\n")

            ml_composite = extract_composite_metrics(composite_data['ml'])
            write_composite_section(f, ml_composite, "ML")

            f.write("\n" + "=" * 120 + "\n")
            f.write(f"V93 SIMULATOR (HF-NOISE REMOVED, LP {cutoff_hz} Hz) - COMPOSITE MATRIX\n")
            f.write("=" * 120 + "\n\n")

            v93_composite = extract_composite_metrics(composite_data['v93'])
            write_composite_section(f, v93_composite, f"V93-LP{cutoff_hz}Hz")

            f.write("\n" + "=" * 120 + "\n")
            f.write(f"MODEL COMPARISON - ML vs V93-HF-Noise-Removed (LP {cutoff_hz} Hz)\n")
            f.write("=" * 120 + "\n\n")

            f.write("F1 Score Comparison:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Category':<15} {'ML F1':<15} {'V93f F1':<15} {'Difference':<15}\n")
            f.write("-" * 60 + "\n")

            for label in ['Air', 'Bld', 'Clt', 'Lch', '2:1:1']:
                ml_f1 = ml_composite.get(label, {}).get('F1', 0)
                v93_f1 = v93_composite.get(label, {}).get('F1', 0)
                diff = v93_f1 - ml_f1
                f.write(f"{label:<15} {ml_f1:<15.4f} {v93_f1:<15.4f} {diff:+.4f}\n")

        print(f"  ✓ Saved: {composite_output}")

    except Exception as e:
        print(f"  ⚠ Error generating composite matrices: {e}")


def extract_composite_metrics(label_rows_dict):
    """Extract TP/TN/FP/FN values from aggregated rows and calculate metrics."""
    composite = {}

    for label, rows in label_rows_dict.items():
        if label == '2:1:1':
            continue

        tp_total = 0
        fn_total = 0
        fp_total = 0
        tn_total = 0

        for row in rows:
            parts = row.split('│')
            if len(parts) >= 5:
                try:
                    tp = int(parts[1].strip())
                    fn = int(parts[2].strip())
                    fp = int(parts[3].strip())
                    tn = int(parts[4].strip())

                    tp_total += tp
                    fn_total += fn
                    fp_total += fp
                    tn_total += tn
                except (ValueError, IndexError):
                    pass

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        specificity = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        denom = ((tp_total + fp_total) * (tp_total + fn_total) * (tn_total + fp_total) * (tn_total + fn_total)) ** 0.5
        mcc = (tp_total * tn_total - fp_total * fn_total) / denom if denom > 0 else 0

        composite[label] = {
            'TP': tp_total, 'FN': fn_total, 'FP': fp_total, 'TN': tn_total,
            'Precision': precision, 'Recall': recall, 'Specificity': specificity,
            'F1': f1, 'MCC': mcc,
        }

    if 'Bld' in composite and 'Clt' in composite and 'Lch' in composite:
        bld_f1 = composite['Bld']['F1']
        clt_f1 = composite['Clt']['F1']
        lch_f1 = composite['Lch']['F1']

        composite_211_f1 = (2 * bld_f1 + clt_f1 + lch_f1) / 4

        composite['2:1:1'] = {
            'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0,
            'Precision': 0.0, 'Recall': 0.0, 'Specificity': 0.0,
            'F1': composite_211_f1, 'MCC': 0.0,
        }

    return composite


def write_composite_section(f, composite_metrics, model_name):
    """Write composite metrics section to file."""
    f.write(f"{'Label':<10} {'TP':>8} {'FN':>8} {'FP':>8} {'TN':>8} {'Precision':>12} {'Recall':>12} {'Specificity':>12} {'F1':>12} {'MCC':>12}\n")
    f.write("-" * 120 + "\n")

    for label in ['Air', 'Bld', 'Clt', 'Lch', '2:1:1']:
        if label in composite_metrics:
            m = composite_metrics[label]
            if label == '2:1:1':
                f.write(f"{label:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} "
                       f"{'N/A':>12} {'N/A':>12} {'N/A':>12} "
                       f"{m['F1']:>12.4f} {'N/A':>12}\n")
            else:
                f.write(f"{label:<10} {m['TP']:>8} {m['FN']:>8} {m['FP']:>8} {m['TN']:>8} "
                       f"{m['Precision']:>12.4f} {m['Recall']:>12.4f} {m['Specificity']:>12.4f} "
                       f"{m['F1']:>12.4f} {m['MCC']:>12.4f}\n")


def main():
    """Main entry point for batch HF-noise-removed v93 processing."""
    parser = argparse.ArgumentParser(description="Batch ML vs HF-Noise-Removed V93 comparison")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip plot generation and only compute confusion matrices")
    parser.add_argument("--force", action="store_true",
                        help="recompute outputs even if result files already exist")
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF_HZ,
                        help=f"low-pass filter cutoff frequency in Hz (default: {DEFAULT_CUTOFF_HZ})")
    args = parser.parse_args()

    cutoff_hz = args.cutoff

    print("=" * 80)
    print("ML vs HF-Noise-Removed V93 Batch Comparison Tool")
    print("=" * 80)
    print(f"V93 simulator will run on LP-filtered impedance (cutoff={cutoff_hz} Hz).")
    print("Cardiac pulse RETAINED, high-frequency noise REMOVED.")
    print(f"ML predictions sourced from: {PARQUET_DIR}/")

    # Create output directories
    print(f"\n📁 Setting up output directories under {OUTPUT_ROOT}/...")
    confusion_matrices_dir, plots_dir, logs_dir = ensure_output_dirs()
    print(f"  ✓ Confusion matrices: {confusion_matrices_dir}")
    print(f"  ✓ Plots: {plots_dir}")
    print(f"  ✓ Logs: {logs_dir}")

    # Initialize database
    print("\n📥 Initializing IDP database...")
    expt_df, event_df, lgtevent_df = db.init(aws_profile_name="default")
    print(f"  ✓ Loaded {len(expt_df)} experiments")

    # Find all parquet files (for experiment list and ML predictions)
    print(f"\n🔍 Scanning for parquet files in {PARQUET_DIR}/...")
    parquet_files = sorted(glob.glob(str(PARQUET_DIR / "*_labeled_segment_denoised_150Hz.parquet")))
    print(f"  ✓ Found {len(parquet_files)} parquet files")

    if len(parquet_files) == 0:
        print("  ✗ No parquet files found!")
        return

    # Process each experiment
    print("\n" + "=" * 80)
    print(f"PROCESSING EXPERIMENTS (HF-Noise-Removed, LP cutoff={cutoff_hz} Hz)")
    print("=" * 80)

    results = {"success": [], "failed": []}

    for parquet_file in parquet_files:
        expt_id = extract_expt_id_from_filename(parquet_file)
        if expt_id is None:
            print(f"⚠ Could not extract experiment ID from {parquet_file}")
            continue

        try:
            success = process_experiment(
                expt_id, parquet_file, expt_df, event_df, lgtevent_df,
                confusion_matrices_dir, plots_dir, cutoff_hz,
                nooplot=args.no_plots,
                force=args.force
            )
            if success:
                results["success"].append(expt_id)
            else:
                results["failed"].append(expt_id)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            traceback.print_exc()
            results["failed"].append(expt_id)

    # Aggregate summaries
    print("\n" + "=" * 80)
    print("AGGREGATING SUMMARIES")
    print("=" * 80)
    aggregate_summaries(confusion_matrices_dir, confusion_matrices_dir, cutoff_hz, expt_ids=results["success"])

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(results['success'])}")
    if results['success']:
        for expt_id in results['success']:
            print(f"    {expt_id}")
    print(f"✗ Failed: {len(results['failed'])}")
    if results['failed']:
        for expt_id in results['failed']:
            print(f"    {expt_id}")

    print(f"\n📁 Output saved to {OUTPUT_ROOT}/")
    print(f"   - Confusion matrices: {OUTPUT_ROOT}/confusion_matrices/")
    print(f"   - Plots: {OUTPUT_ROOT}/plots/")
    print(f"   - ML aggregated summary: {OUTPUT_ROOT}/confusion_matrices/ML_MODEL_AGGREGATED_SUMMARY.txt")
    print(f"   - V93-HF-noise-removed summary: {OUTPUT_ROOT}/confusion_matrices/V93_HFNOISE_REMOVED_AGGREGATED_SUMMARY.txt")
    print(f"   - Combined summary: {OUTPUT_ROOT}/confusion_matrices/COMBINED_ML_V93_HFNOISE_REMOVED_SUMMARY.txt")
    print(f"   - Composite matrices: {OUTPUT_ROOT}/confusion_matrices/COMPOSITE_CONFUSION_MATRICES.txt")
    print(f"\n✅ Batch HF-noise-removed v93 processing complete! (cutoff={cutoff_hz} Hz)")


if __name__ == "__main__":
    main()
