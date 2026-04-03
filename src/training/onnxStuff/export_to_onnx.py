# export_to_onnx.py
# Separate script to load .pt, quantize, and export to ONNX

import torch
import torch.nn as nn
from pathlib import Path
import torch.quantization as quant
import sys
# from pathlib import Path
from gru_torch_V5 import ClotGRU  # Import your ClotGRU class

# PROJECT_ROOT = Path(__file__).resolve().parents[2]  # goes up from src/data/ → PyTorch_3
# sys.path.insert(0, str(PROJECT_ROOT))

# ================= CONFIG =================
# SCRIPT_DIR    = Path(__file__).resolve().parent
# PROJECT_ROOT  = SCRIPT_DIR.parent.parent
PT_FILE_PATH =   "clot_gru_trained.pt"  # Change to your .pt path
ONNX_FLOAT_PATH = "clot_gru_float.onnx"
ONNX_QUANTIZED_PATH = "clot_gru_quantized.onnx"
# =========================================

def load_model(pt_path):
    model = ClotGRU()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    return model

def quantize_model(model):
    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},  # Quantize Linear and GRU layers
        dtype=torch.qint8
    )
    return quantized_model

def export_to_onnx(model, onnx_path, input_size=40):
    dummy_input = torch.randn(1, input_size)  # Batch=1, 40 features

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=9,  # Compatible with most embedded tools
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported to ONNX: {onnx_path}")

def main():
    print(f"Loading model from: {PT_FILE_PATH}")
    float_model = load_model(PT_FILE_PATH)

    # Export float model
    export_to_onnx(float_model, ONNX_FLOAT_PATH)

    # Quantize and export quantized model
    # print("\nQuantizing model...")
    # quantized_model = quantize_model(float_model)

    # export_to_onnx(quantized_model, ONNX_QUANTIZED_PATH)

if __name__ == "__main__":
    main()