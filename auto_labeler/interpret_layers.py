"""
UNet Layer Interpretability Visualization.

Shows what each encoder/decoder level learns by visualizing activations
aligned with the input signal and labels.

Usage:
    python auto_labeler/interpret_layers.py [parquet_file] [--time_start SEC]
    python auto_labeler/interpret_layers.py [parquet_file] --full   # entire signal
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

from auto_labeler import config as cfg
from auto_labeler.predict import load_model
from auto_labeler.model import UNet1D


def extract_layer_activations(model: UNet1D, chunk: torch.Tensor) -> dict:
    """Run a single chunk through the model and capture all intermediate activations.

    Returns dict with keys:
        input, enc_1..enc_5 (skip features), bottleneck, dec_1..dec_5, output
    Each value is (channels, spatial_dim) numpy array.
    """
    model.eval()
    activations = {}

    with torch.no_grad():
        x = chunk.unsqueeze(0)  # (1, C, L)
        activations["input"] = x.squeeze(0).cpu().numpy()

        # Encoder path
        skips = []
        for i, (encoder, drop) in enumerate(zip(model.encoders, model.enc_dropouts)):
            x, features = encoder(x)
            x = drop(x)
            skips.append(features)
            activations[f"enc_{i+1}"] = features.squeeze(0).cpu().numpy()

        # Bottleneck
        x = model.bottleneck(x)
        x = model.bottleneck_dropout(x)
        activations["bottleneck"] = x.squeeze(0).cpu().numpy()

        # Decoder path
        for i, (decoder, skip) in enumerate(zip(model.decoders, reversed(skips))):
            x = decoder(x, skip)
            activations[f"dec_{i+1}"] = x.squeeze(0).cpu().numpy()

        # Output
        out = model.head(x)
        activations["output"] = out.squeeze(0).cpu().numpy()
        activations["softmax"] = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()

    return activations


def plot_layer_activations(activations: dict, time_sec: np.ndarray,
                           labels: np.ndarray = None, study_id: str = ""):
    """Create interactive Plotly visualization of layer activations."""

    # Layers to show (encoder → bottleneck → decoder)
    layer_order = (
        ["input"] +
        [f"enc_{i}" for i in range(1, 6)] +
        ["bottleneck"] +
        [f"dec_{i}" for i in range(1, 6)] +
        ["softmax"]
    )
    layer_names = {
        "input": "Input (1ch, 4096)",
        "enc_1": "Encoder 1 (32ch, 2048)",
        "enc_2": "Encoder 2 (64ch, 1024)",
        "enc_3": "Encoder 3 (128ch, 512)",
        "enc_4": "Encoder 4 (256ch, 256)",
        "enc_5": "Encoder 5 (512ch, 128)",
        "bottleneck": "Bottleneck (512ch, 128)",
        "dec_1": "Decoder 1 (256ch, 256)",
        "dec_2": "Decoder 2 (128ch, 512)",
        "dec_3": "Decoder 3 (64ch, 1024)",
        "dec_4": "Decoder 4 (32ch, 2048)",
        "dec_5": "Decoder 5 (32ch, 4096)",
        "softmax": "Output Softmax (3ch, 4096)",
    }

    n_rows = len(layer_order) + (1 if labels is not None else 0)
    row_heights = [0.08] * len(layer_order)
    if labels is not None:
        row_heights = [0.05] + row_heights  # GT labels row at top

    subtitles = []
    if labels is not None:
        subtitles.append("Ground Truth Labels")
    for key in layer_order:
        act = activations[key]
        subtitles.append(f"{layer_names.get(key, key)} — shape {act.shape}")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.005,
        subplot_titles=subtitles,
    )

    row = 1

    # GT labels as colored band
    if labels is not None:
        colors = {0: "rgba(60,179,113,0.5)", 1: "rgba(220,60,60,0.5)", 2: "rgba(70,130,180,0.4)"}
        # Add invisible trace for x-range
        fig.add_trace(go.Scatter(x=[time_sec[0], time_sec[-1]], y=[0.5, 0.5],
                                 mode="lines", line=dict(width=0), showlegend=False),
                      row=row, col=1)
        # Add colored rectangles
        changes = np.where(np.diff(labels) != 0)[0] + 1
        segments = np.split(np.arange(len(labels)), changes)
        for seg in segments:
            if len(seg) == 0:
                continue
            lbl = labels[seg[0]]
            fig.add_shape(type="rect", x0=time_sec[seg[0]], x1=time_sec[seg[-1]],
                          y0=0, y1=1, fillcolor=colors[lbl], line_width=0,
                          xref=f"x", yref=f"y", layer="below")
        fig.update_yaxes(range=[0, 1], showticklabels=False, row=row, col=1)
        row += 1

    # Plot each layer's activation
    for key in layer_order:
        act = activations[key]  # (channels, spatial_dim)
        n_ch, spatial = act.shape

        # Create time axis for this layer's resolution
        layer_time = np.linspace(time_sec[0], time_sec[-1], spatial)

        if key == "softmax":
            # Show all 3 class probabilities
            class_names = cfg.CLASS_NAMES
            class_colors = ["seagreen", "crimson", "steelblue"]
            for c in range(3):
                fig.add_trace(go.Scattergl(
                    x=layer_time, y=act[c],
                    mode="lines", line=dict(width=1.5, color=class_colors[c]),
                    name=class_names[c], showlegend=(row == n_rows),
                ), row=row, col=1)
            fig.update_yaxes(range=[0, 1], title_text="P", row=row, col=1)
        elif key == "input":
            # Show raw input signal
            fig.add_trace(go.Scattergl(
                x=layer_time, y=act[0],
                mode="lines", line=dict(width=0.8, color="black"),
                name="Input", showlegend=False,
            ), row=row, col=1)
        else:
            # Show mean, max, and a few top-activating channels
            mean_act = act.mean(axis=0)
            max_act = act.max(axis=0)
            min_act = act.min(axis=0)

            # Mean activation (shows overall response)
            fig.add_trace(go.Scattergl(
                x=layer_time, y=mean_act,
                mode="lines", line=dict(width=1.2, color="navy"),
                name=f"mean", showlegend=False,
            ), row=row, col=1)

            # Max activation envelope (shows strongest feature response)
            fig.add_trace(go.Scattergl(
                x=layer_time, y=max_act,
                mode="lines", line=dict(width=0.5, color="rgba(255,140,0,0.6)"),
                name=f"max", showlegend=False,
            ), row=row, col=1)

            # Min activation envelope
            fig.add_trace(go.Scattergl(
                x=layer_time, y=min_act,
                mode="lines", line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
                name=f"min", showlegend=False,
            ), row=row, col=1)

        row += 1

    fig.update_layout(
        height=200 * n_rows,
        width=1400,
        title_text=f"UNet Layer Activations — {study_id}" if study_id else "UNet Layer Activations",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Time (sec)", row=n_rows, col=1)

    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize UNet layer activations")
    parser.add_argument("input", nargs="?", default=None,
                        help="Parquet file (default: first file in test_data/)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--time_start", type=float, default=None,
                        help="Start time in seconds for the chunk (default: first clot event)")
    parser.add_argument("--full", action="store_true",
                        help="Process entire signal (tiles into chunks and stitches)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: auto-generated)")
    args = parser.parse_args()

    # Find checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        candidates = [
            script_dir / "checkpoints" / "best_model.pt",
            Path("auto_labeler/checkpoints/best_model.pt"),
        ]
        for c in candidates:
            if c.exists():
                checkpoint = str(c)
                break
    if not checkpoint:
        print("ERROR: No checkpoint found. Use --checkpoint")
        return

    # Find input file
    input_path = args.input
    if not input_path:
        test_dirs = [script_dir / "test_data", script_dir / "test_data_8"]
        for d in test_dirs:
            if d.exists():
                parquets = sorted(d.glob("*_labeled_segment.parquet"))
                if parquets:
                    input_path = str(parquets[0])
                    break
    if not input_path:
        print("ERROR: No input file. Provide a parquet file as argument.")
        return

    print(f"Checkpoint: {checkpoint}")
    print(f"Input: {input_path}")

    # Load model
    device = torch.device("cpu")  # CPU is fine for single-chunk visualization
    model, multichannel = load_model(checkpoint, device)
    print(f"Model loaded (multichannel={multichannel})")

    # Load data
    df = pd.read_parquet(input_path)
    resistance = df["magRLoadAdjusted"].values.astype(np.float32)
    has_labels = "label" in df.columns
    labels = df["label"].values if has_labels else None

    # Z-normalize (single channel)
    mean, std = resistance.mean(), resistance.std() + 1e-8
    normalized = (resistance - mean) / std

    # Pick chunk(s)
    chunk_size = cfg.CHUNK_SIZE

    if args.full:
        # --- Full signal: tile into non-overlapping chunks and stitch ---
        n_samples = len(normalized)
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        # Pad signal to exact multiple of chunk_size
        pad_len = n_chunks * chunk_size - n_samples
        if pad_len > 0:
            normalized = np.concatenate([normalized, np.zeros(pad_len, dtype=np.float32)])
            if has_labels:
                labels = np.concatenate([labels, np.full(pad_len, 0, dtype=labels.dtype)])

        print(f"Full signal: {n_samples} samples, {n_chunks} chunks")

        # Process each chunk and collect activations
        all_activations = None
        for ci in range(n_chunks):
            start = ci * chunk_size
            chunk_data = normalized[start:start + chunk_size]
            chunk_tensor = torch.tensor(chunk_data, dtype=torch.float32).unsqueeze(0)
            act = extract_layer_activations(model, chunk_tensor)

            if all_activations is None:
                all_activations = {k: [v] for k, v in act.items()}
            else:
                for k, v in act.items():
                    all_activations[k].append(v)

            if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
                print(f"  Processed chunk {ci+1}/{n_chunks}")

        # Stitch: concatenate along spatial axis (axis=1 for (channels, spatial))
        activations = {k: np.concatenate(v, axis=1) for k, v in all_activations.items()}

        # Trim padded portion from full-resolution layers
        if pad_len > 0:
            for k in activations:
                spatial = activations[k].shape[1]
                # Compute the expected unpadded spatial length
                ratio = spatial / (n_chunks * chunk_size)
                trim = int(n_samples * ratio)
                activations[k] = activations[k][:, :trim]
            labels = labels[:n_samples] if has_labels else None

        # Time axis
        if "timeInMS" in df.columns:
            time_sec = df["timeInMS"].values[:n_samples] / 1000.0
        else:
            time_sec = np.arange(n_samples) / cfg.SAMPLING_RATE_HZ

        chunk_labels = labels
        print(f"Signal: t={time_sec[0]:.1f}s to {time_sec[-1]:.1f}s")

    else:
        # --- Single chunk mode ---
        if args.time_start is not None:
            sample_idx = int(args.time_start * cfg.SAMPLING_RATE_HZ)
        elif has_labels:
            clot_mask = labels == 1
            if clot_mask.any():
                clot_start = np.where(clot_mask)[0][0]
                sample_idx = max(0, clot_start - chunk_size // 4)
            else:
                sample_idx = len(resistance) // 4
        else:
            sample_idx = len(resistance) // 4

        sample_idx = min(sample_idx, len(resistance) - chunk_size)
        sample_idx = max(0, sample_idx)

        chunk_data = normalized[sample_idx:sample_idx + chunk_size]
        chunk_labels = labels[sample_idx:sample_idx + chunk_size] if has_labels else None

        if "timeInMS" in df.columns:
            time_sec = df["timeInMS"].values[sample_idx:sample_idx + chunk_size] / 1000.0
        else:
            time_sec = np.arange(chunk_size) / cfg.SAMPLING_RATE_HZ + sample_idx / cfg.SAMPLING_RATE_HZ

        print(f"Chunk: samples {sample_idx} to {sample_idx + chunk_size} "
              f"(t={time_sec[0]:.1f}s to {time_sec[-1]:.1f}s)")

        chunk_tensor = torch.tensor(chunk_data, dtype=torch.float32).unsqueeze(0)
        activations = extract_layer_activations(model, chunk_tensor)

    # Print summary
    print(f"\nLayer activation shapes:")
    for key in ["input", "enc_1", "enc_2", "enc_3", "enc_4", "enc_5",
                "bottleneck", "dec_1", "dec_2", "dec_3", "dec_4", "dec_5", "softmax"]:
        act = activations[key]
        print(f"  {key:12s}: {str(act.shape):15s}  "
              f"mean={act.mean():.3f}  std={act.std():.3f}  "
              f"min={act.min():.3f}  max={act.max():.3f}")

    # Plot
    study_id = Path(input_path).stem.replace("_labeled_segment", "")
    fig = plot_layer_activations(activations, time_sec, chunk_labels, study_id)

    # Save
    output_path = args.output
    if not output_path:
        output_path = str(script_dir / f"layer_activations_{study_id}.html")
    fig.write_html(output_path)
    print(f"\nSaved: {output_path}")
    print("Open in browser to explore — zoom into boundaries to see how layers respond.")


if __name__ == "__main__":
    main()
