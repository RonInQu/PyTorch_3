# export_to_onnx.py
# Separate script to load .pt, quantize, and export to ONNX

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
# =========================================


class ClotGRUForExport(nn.Module):
    """Wrapper that returns only logits (no hidden state) for clean ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x, None)
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
    wrapper = ClotGRUForExport(model)
    wrapper.eval()

    dummy_input = torch.randn(1, seq_len, input_size)

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"Exported to ONNX: {onnx_path}")
    print(f"  input_size={input_size}, seq_len={seq_len}")

def main():
    print(f"Loading model from: {PT_FILE_PATH}")
    print(f"  active_dim={active_dim}, SEQ_LEN={SEQ_LEN}")
    float_model = load_model(PT_FILE_PATH)

    export_to_onnx(float_model, ONNX_FLOAT_PATH)

    # Quantized version later if needed

if __name__ == "__main__":
    main()