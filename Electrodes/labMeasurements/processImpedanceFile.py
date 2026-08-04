import os
import re

def process_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    in_trace_b = False
    new_content = []
    
    for line in content:
        stripped = line.strip()
        
        if stripped == '"TRACE: B"':
            in_trace_b = True
            new_content.append(line)
            continue
        elif stripped == '"TRACE: A"':
            in_trace_b = False
            new_content.append(line)
            continue
        
        if in_trace_b and re.match(r'^\s*[\d.eE+-]+\s', line):
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 3:
                try:
                    freq = parts[0]
                    real_val = float(parts[1])
                    imag_val = parts[2]
                    new_real = real_val - 90
                    new_line = f"{freq}\t{new_real:.6e}\t{imag_val}\n"
                    new_content.append(new_line)
                    continue
                except ValueError:
                    pass
        
        new_content.append(line)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)

# ================== CONFIG ==================
base_dir = r'C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Electrodes\labMeasurements'
# ===========================================

files = ['FN_Clot2A.TXT', 'FN_Clot2B.TXT', 'FN_Clot4A.TXT', 'FN_Clot4B.TXT']

for fname in files:
    input_path = os.path.join(base_dir, fname)
    output_path = os.path.join(base_dir, fname.replace('.TXT', '.txt'))
    
    if os.path.exists(input_path):
        process_file(input_path, output_path)
        print(f"✅ Processed: {fname} → {os.path.basename(output_path)}")
    else:
        print(f"⚠️  File not found: {fname}")