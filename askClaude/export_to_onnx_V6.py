# export_to_onnx_V6.py
# Separate script to load .pt, quantize, and export to ONNX
#
# STM32Cube.AI compatibility notes:
#   - The nn.GRU ONNX operator has multi-output tuples that STM32Cube.AI
#     cannot parse ("Tuple out of range" error).  We unroll the GRU into
#     a GRUCell loop so the ONNX graph contains only matmul/sigmoid/tanh.
#   - Opset kept at 13 (max widely supported by STM32Cube.AI 8.x/9.x).
#   - All shapes are fully static (batch=1) — no dynamic_axes.

import torch
import torch.nn as nn
from pathlib import Path
import torch.quantization as quant
import sys
from gru_torch_V6 import ClotGRU, active_dim, SEQ_LEN

# ================= CONFIG =================
PT_FILE_PATH =   "clot_gru_trained.pt"
ONNX_FLOAT_PATH = "clot_gru_float.onnx"
ONNX_QUANTIZED_PATH = "clot_gru_quantized.onnx"
OPSET_VERSION = 13   # STM32Cube.AI: use 13-15 (18 causes "Tuple out of range")
# =========================================


class ClotGRUUnrolled(nn.Module):
    """STM32Cube.AI-friendly model: GRU unrolled into a GRUCell loop.

    Replaces the nn.GRU ONNX operator (which has multi-output tuples
    that STM32Cube.AI can't parse) with explicit GRUCell steps that
    export as simple matmul/sigmoid/tanh ops.

    Weights are copied from the trained ClotGRU — no retraining needed.
    """
    def __init__(self, trained_model: ClotGRU):
        super().__init__()
        gru = trained_model.gru
        self.hidden_size = gru.hidden_size

        # Create a GRUCell with the same weights as the nn.GRU layer
        self.cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        self.cell.weight_ih = gru.weight_ih_l0
        self.cell.weight_hh = gru.weight_hh_l0
        self.cell.bias_ih   = gru.bias_ih_l0
        self.cell.bias_hh   = gru.bias_hh_l0

        # Copy the FC head
        self.fc1 = trained_model.fc1
        self.fc2 = trained_model.fc2

    def forward(self, x):
        # x: (1, seq_len, input_size) — batch is always 1 for STM32
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Unrolled GRU: step through each timestep explicitly
        for t in range(x.size(1)):
            h = self.cell(x[:, t, :], h)

        # FC head (same as ClotGRU.forward)
        out = torch.relu(self.fc1(h))
        logits = self.fc2(out)
        return logits


def load_model(pt_path):
    model = ClotGRU()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    return model

def quantize_model(model):
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8
    )
    return quantized_model

def export_to_onnx(model, onnx_path, input_size=active_dim, seq_len=SEQ_LEN):
    unrolled = ClotGRUUnrolled(model)
    unrolled.eval()

    # Verify numerical equivalence before exporting
    _verify_equivalence(model, unrolled, input_size, seq_len)

    dummy_input = torch.randn(1, seq_len, input_size)

    torch.onnx.export(
        unrolled,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        # No dynamic_axes — STM32 needs fully static shapes (batch=1)
    )
    print(f"Exported to ONNX: {onnx_path}")
    print(f"  input_size={input_size}, seq_len={seq_len}, opset={OPSET_VERSION}")

    # Optional: simplify the graph (removes redundant ops, folds constants)
    try:
        import onnx
        from onnxsim import simplify
        model_onnx = onnx.load(onnx_path)
        model_simp, ok = simplify(model_onnx)
        if ok:
            onnx.save(model_simp, onnx_path)
            print(f"  ONNX graph simplified successfully")
        else:
            print(f"  Warning: onnx-simplifier could not simplify (using original)")
    except ImportError:
        print(f"  Tip: pip install onnx onnxsim  for graph simplification")


@torch.no_grad()
def _verify_equivalence(original, unrolled, input_size, seq_len, atol=1e-5):
    """Verify the unrolled model produces identical logits to the original."""
    original.eval()
    unrolled.eval()
    x = torch.randn(1, seq_len, input_size)
    logits_orig, _ = original(x, None)
    logits_unrolled = unrolled(x)
    max_diff = (logits_orig - logits_unrolled).abs().max().item()
    if max_diff > atol:
        print(f"  WARNING: max logit difference = {max_diff:.2e} (threshold={atol:.0e})")
        print(f"           Unrolled model may not match original!")
        sys.exit(1)
    else:
        print(f"  Equivalence verified: max diff = {max_diff:.2e} (< {atol:.0e})")

def main():
    print(f"Loading model from: {PT_FILE_PATH}")
    print(f"  active_dim={active_dim}, SEQ_LEN={SEQ_LEN}, opset={OPSET_VERSION}")
    float_model = load_model(PT_FILE_PATH)

    export_to_onnx(float_model, ONNX_FLOAT_PATH)

    # Quantized version later if needed

if __name__ == "__main__":
    main()