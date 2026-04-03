#!/usr/bin/env python3
"""
Downgrade ONNX model IR version from 10 to 9 for X-CUBE-AI compatibility.
X-CUBE-AI 10.2.0 supports up to IR version 9.
"""

import sys
import os

try:
    import onnx
except ImportError:
    print("ERROR: onnx package not installed. Run: pip install onnx")
    sys.exit(1)

def downgrade_onnx_ir(input_path: str, output_path: str = None, target_ir: int = 9):
    """Downgrade ONNX model IR version."""
    
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading: {input_path}")
    model = onnx.load(input_path)
    
    original_ir = model.ir_version
    print(f"Original IR version: {original_ir}")
    
    if original_ir <= target_ir:
        print(f"Model already at IR version {original_ir} (<= {target_ir}). No change needed.")
        return
    
    # Downgrade IR version
    model.ir_version = target_ir
    print(f"Downgraded to IR version: {target_ir}")
    
    # Generate output path if not specified
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_ir{target_ir}{ext}"
    
    # Save the modified model
    onnx.save(model, output_path)
    print(f"Saved to: {output_path}")
    
    # Verify the saved model
    try:
        onnx.checker.check_model(output_path)
        print("Model validation: PASSED")
    except Exception as e:
        print(f"Model validation WARNING: {e}")
        print("The model may still work with X-CUBE-AI despite this warning.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python downgrade_onnx_ir.py <input.onnx> [output.onnx]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    downgrade_onnx_ir(input_file, output_file)