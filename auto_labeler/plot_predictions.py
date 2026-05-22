"""Plot predicted vs ground truth labels at full sample resolution.

Overlays colored labels on the signal so you can zoom to see exact boundaries.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# --- Config ---
COLORS = {0: "steelblue", 1: "crimson", 2: "seagreen"}
FILL_COLORS = {0: "rgba(70,130,180,0.25)", 1: "rgba(220,60,60,0.25)", 2: "rgba(60,179,113,0.25)"}
LABEL_NAMES = {0: "Blood", 1: "Clot", 2: "Wall"}
SAMPLE_RATE = 150  # Hz


def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    df["time_sec"] = (df["timeInMS"] - df["timeInMS"].iloc[0]) / 1000.0
    return df


def color_signal_by_label(fig, df, label_col, row, signal_col="magRLoadAdjusted"):
    """Draw the signal colored by label — one trace per contiguous segment."""
    labels = df[label_col].values
    time = df["time_sec"].values
    signal = df[signal_col].values

    # Find change points
    changes = np.where(np.diff(labels) != 0)[0] + 1
    segments = np.split(np.arange(len(labels)), changes)

    for i, seg in enumerate(segments):
        if len(seg) == 0:
            continue
        lbl = labels[seg[0]]
        # Extend by 1 sample into next segment for continuity
        end = min(seg[-1] + 2, len(time))
        idx = slice(seg[0], end)
        fig.add_trace(
            go.Scattergl(
                x=time[idx],
                y=signal[idx],
                mode="lines",
                line=dict(width=1.5, color=COLORS[lbl]),
                showlegend=False,
                hoverinfo="x+y",
            ),
            row=row, col=1,
        )


def add_label_step(fig, df, label_col, row):
    """Draw label as a step function (0=blood, 1=clot, 2=wall) for precise boundary view."""
    time = df["time_sec"].values
    labels = df[label_col].values
    fig.add_trace(
        go.Scattergl(
            x=time,
            y=labels,
            mode="lines",
            line=dict(width=1.5, color="black", shape="hv"),  # step function
            showlegend=False,
            hovertemplate="t=%{x:.3f}s label=%{y}<extra></extra>",
        ),
        row=row, col=1,
    )


def make_plot(parquet_path):
    df = load_data(parquet_path)
    study_id = os.path.basename(parquet_path).split("_")[0]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.35, 0.15, 0.15],
        vertical_spacing=0.04,
        subplot_titles=[
            f"Signal colored by Ground Truth — {study_id}",
            "Signal colored by Predicted",
            "GT label (step)",
            "Predicted label (step)",
        ],
    )

    # Row 1: Signal colored by GT
    color_signal_by_label(fig, df, "label", row=1)

    # Row 2: Signal colored by Predicted
    color_signal_by_label(fig, df, "predicted_label", row=2)

    # Row 3: GT step function
    add_label_step(fig, df, "label", row=3)

    # Row 4: Predicted step function
    add_label_step(fig, df, "predicted_label", row=4)

    # Legend entries
    for lbl, color in COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(width=4, color=color),
                name=LABEL_NAMES[lbl],
                showlegend=True,
            )
        )

    # Layout
    fig.update_layout(
        height=850,
        title_text=f"Boundary Analysis: {study_id}",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Time (sec)", row=4, col=1)
    fig.update_yaxes(title_text="magR", row=1, col=1)
    fig.update_yaxes(title_text="magR", row=2, col=1)
    fig.update_yaxes(
        title_text="Label", row=3, col=1,
        tickvals=[0, 1, 2], ticktext=["Blood", "Clot", "Wall"],
        range=[-0.2, 2.2],
    )
    fig.update_yaxes(
        title_text="Label", row=4, col=1,
        tickvals=[0, 1, 2], ticktext=["Blood", "Clot", "Wall"],
        range=[-0.2, 2.2],
    )

    # Save HTML
    out_path = parquet_path.replace(".parquet", ".html")
    fig.write_html(out_path)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "auto_labeler/8ECEADA6_labeled_segment.parquet"
    make_plot(path)
