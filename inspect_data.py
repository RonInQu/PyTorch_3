import pandas as pd
import os

data_dir = r"c:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\training_data"
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
if not files:
    print("No parquet files found.")
    exit()

print(f"Total files: {len(files)}")
print(f"First 5: {files[:5]}")

# Load one file and inspect
df = pd.read_parquet(os.path.join(data_dir, files[0]))
print(f"\nColumns: {list(df.columns)}")
print(f"Shape: {df.shape}")
print(f"Dtypes:\n{df.dtypes}")
print(f"\nLabel distribution:\n{df['label'].value_counts().sort_index()}")
print(f"\nR stats:\n{df['magRLoadAdjusted'].describe()}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nSampling: dt = {df['timeInMS'].diff().median():.2f} ms")

# Check a second file for variety
if len(files) > 10:
    df2 = pd.read_parquet(os.path.join(data_dir, files[10]))
    print(f"\n--- File: {files[10]} ---")
    print(f"Shape: {df2.shape}")
    print(f"Label distribution:\n{df2['label'].value_counts().sort_index()}")
    print(f"R range: {df2['magRLoadAdjusted'].min():.1f} - {df2['magRLoadAdjusted'].max():.1f}")
else:
    print("\nSkipping second file check (less than 11 files).")
