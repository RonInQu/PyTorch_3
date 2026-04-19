import pandas as pd
import numpy as np
import os

base = r'C:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Working\Ronald Kurnik\merged_expts_with_events_April6\event_files\processedResults'

# New file (04.19) - from today's run
new_path = None
for subfolder in ['training', 'testing']:
    p = os.path.join(base, subfolder, '33CFB812_labeled_segment.parquet')
    if os.path.exists(p):
        new_path = p
        print(f"New file found in: {subfolder}")
        break

# Old file (04.07) - from previous pipeline
old_path = None
for folder in ['training_data', 'test_data']:
    p = os.path.join(folder, '33CFB812_labeled_segment.parquet')
    if os.path.exists(p):
        old_path = p
        print(f"Old file found in: {folder}")
        break

dfs = {}
for label, path in [('new_04.19', new_path), ('old_04.07', old_path)]:
    if path and os.path.exists(path):
        df = pd.read_parquet(path)
        dfs[label] = df
        print(f"\n=== {label} ({path}) ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"\nHead:\n{df.head(10)}")
        print(f"\nTail:\n{df.tail(5)}")
        print(f"\nDescribe:\n{df.describe()}")
        if 'label' in df.columns:
            print(f"\nLabel distribution: {df['label'].value_counts().sort_index().to_dict()}")
        if 'da_label' in df.columns:
            print(f"da_label distribution: {df['da_label'].value_counts().sort_index().to_dict()}")
        print(f"\nIndex info: type={type(df.index)}, name={df.index.name}, range={df.index.min()}-{df.index.max()}")
    else:
        print(f"\n{label}: NOT FOUND")

if len(dfs) == 2:
    old = dfs['old_04.07']
    new = dfs['new_04.19']
    print("\n\n=== COMPARISON ===")
    print(f"Shape: old={old.shape}, new={new.shape}")
    print(f"Columns: old={list(old.columns)}, new={list(new.columns)}")
    
    common_cols = set(old.columns) & set(new.columns)
    print(f"Common columns: {common_cols}")
    print(f"Only in old: {set(old.columns) - set(new.columns)}")
    print(f"Only in new: {set(new.columns) - set(old.columns)}")
    
    if old.shape[0] == new.shape[0]:
        for col in sorted(common_cols):
            if old[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                diff = (old[col] - new[col]).abs()
                print(f"\n  {col}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, exact_match={diff.sum()==0}")
            else:
                match = (old[col] == new[col]).all()
                print(f"\n  {col}: exact_match={match}")
