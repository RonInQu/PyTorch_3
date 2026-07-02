import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== YOUR PATH ======================
csv_path = r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3\Comsol\Geom25\5kHz_Waveform.csv"

# Output folder = same folder as the input file
output_folder = os.path.dirname(csv_path)
os.makedirs(output_folder, exist_ok=True)

table_path = os.path.join(output_folder, "small_table_560mV.csv")
plot_path  = os.path.join(output_folder, "waveform_plot_560mV.png")
# =======================================================

def analyze_5kHz_waveform(csv_path: str, target_pp_mV: float = 560, target_samples: int = 100):
    """
    Load 5 kHz waveform, scale to scope measurement (560 mV p-p),
    return small descriptive table + summary stats + plot.
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=12, header=None, names=["Time", "Voltage"])
    
    # Scale voltages so peak-to-peak matches your scope reading
    actual_pp = df["Voltage"].max() - df["Voltage"].min()
    scale = (target_pp_mV / 1000) / actual_pp
    df["Voltage_Adjusted"] = df["Voltage"] * scale
    
    # Create small evenly-spaced table (100 points)
    indices = np.linspace(0, len(df) - 1, target_samples, dtype=int)
    df_small = df.iloc[indices][["Time", "Voltage_Adjusted"]].reset_index(drop=True)
    df_small.columns = ["Time (s)", "Voltage (V)"]
    
    # Summary statistics
    stats = {
        "Duration_ms": round((df["Time"].max() - df["Time"].min()) * 1000, 2),
        "Min_V": round(df["Voltage_Adjusted"].min(), 3),
        "Max_V": round(df["Voltage_Adjusted"].max(), 3),
        "Mean_V": round(df["Voltage_Adjusted"].mean(), 4),
        "RMS_V": round(np.sqrt(np.mean(df["Voltage_Adjusted"]**2)), 4),
        "Peak_to_Peak_V": round(target_pp_mV / 1000, 3),
        "Original_p-p_V": round(actual_pp, 3),
        "Scale_Factor": round(scale, 3),
        "Num_Samples_Original": len(df)
    }
    
    print("\n=== Adjusted 5 kHz Waveform Summary (560 mV p-p) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Save small table
    df_small.to_csv(table_path, index=False)
    print(f"\nSmall table saved to: {table_path}")
    
    # Optional: Save plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["Time"] * 1000, df["Voltage_Adjusted"], 
             linewidth=0.6, color="#1f77b4", alpha=0.85)
    plt.title(f"5 kHz Waveform — Adjusted to {target_pp_mV} mV Peak-to-Peak")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    
    return df_small, pd.Series(stats)


# ====================== RUN ======================
if __name__ == "__main__":
    df_small, stats = analyze_5kHz_waveform(
        csv_path=csv_path,
        target_pp_mV=560,      # ← matches your scope measurement
        target_samples=100
    )