def classify_media_state(f_r, Q, f_dry_cal=9999995.0):
    """
    Real-time classification logic calibrated strictly to simulated data:
    
    Metrics:
    - Air:        delta_f = 0 Hz,        Q = 124,937
    - Blood:      delta_f = -1,000.5 Hz, Q = 6,325
    - Clot:       delta_f = -4,982.5 Hz, Q = 1,456
    - Wall:       delta_f = -19,930 Hz,  Q = 3,807
    """
    # Calculate frequency shift from dry baseline
    delta_f = f_r - f_dry_cal
    
    # 1. Air / Dry Baseline Check
    # Frequency is stable within 100 Hz of baseline and Q is extremely high
    if abs(delta_f) < 100 and Q > 50000:
        return "AIR / DRY BASELINE"
        
    # 2. Vessel Wall Contact Check (Safety Trigger)
    # The vessel wall causes a massive frequency drop past -15,000 Hz, 
    # but maintains a healthy elastic Q-factor well above the clot layer.
    if delta_f < -15000:
        return "CRITICAL: VESSEL WALL CONTACT"
        
    # 3. Clot vs. Flowing Blood Check
    # A clot drops the frequency significantly more than liquid blood (~5 kHz vs ~1 kHz)
    # AND drops the Q-factor below a clear 3,000 threshold due to viscoelastic damping.
    if Q < 3000:
        return "CLOT DETECTED (ENGULFED)"
    else:
        return "BLOOD FLOWING (CLEAR)"

# --- Quick verification block using your exact table values ---
test_cases = {
    "Air Sample":       (9999995.0, 124937.4),
    "Blood Sample":     (9998994.5, 6325.3),
    "Clot Sample":      (9995012.5, 1456.3),
    "Vessel Wall":      (9980065.0, 3807.3)
}

print("Executing classification verification:")
print("-" * 55)
for name, (f_r, Q) in test_cases.items():
    result = classify_media_state(f_r, Q)
    print(f"{name:<15} -> Result: {result}")