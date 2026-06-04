"""Add Manual_GT column to Centennial 054 and 055 parquets from manual annotations."""
import pandas as pd
import numpy as np

# Label mapping: string -> numeric code
# 4=SALINE_BLOOD, 5=CLOT, 9=WALL_LATCH, 2=CMS_ERROR(unknown), 8=FLUID_INJECTION(contrast), 0=OFF, 6=AIR
label_to_num = {
    'blood': 4,
    'clot': 5,
    'wall': 9,
    'unknown': 2,
    'contrast': 8,
    'entry_exit': 0,
    'saline_prep': 4,
    'air': 6,
    'clot_wall': 2,  # ambiguous -> unknown
}

cases = [
    {
        'name': 'Centennial 054',
        'parquet': '2026-05-13 220-054 Centennial_state.parquet',
        'csv': r"c:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Pipeline\Staging_IDE_Powered\2026-05-13 220-054 Centennial\events220-054 Centennial.csv",
    },
    {
        'name': 'Centennial 055',
        'parquet': '2026-05-13 220-055 Centennial_state.parquet',
        'csv': r"c:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Pipeline\Staging_IDE_Powered\2026-05-13 220-055 Centennial\events220-055 Centennial.csv",
    },
]


def classify_label(t_str):
    """Map annotation type string to a tissue label."""
    t_clean = t_str.strip().replace('\u200b', '').replace('\u200c', '')
    if 'TRACKING' in t_clean:
        return 'blood'
    elif 'WALL_LATCH' in t_clean:
        return 'wall'
    elif 'CLOT_IN_AREA' in t_clean:
        return 'clot'
    elif 'CLOT_WALL' in t_clean:
        return 'clot_wall'
    elif 'CLOT' in t_clean and 'WALL' not in t_clean:
        return 'clot'
    elif 'CONTRAST' in t_clean:
        return 'contrast'
    elif 'UNKNOWN' in t_clean or 'UNGRADABLE' in t_clean:
        return 'unknown'
    elif 'ENTRY_EXIT' in t_clean:
        return 'entry_exit'
    elif 'START_HERE' in t_clean:
        return 'start_marker'
    elif 'END_HERE' in t_clean:
        return 'end_marker'
    elif 'SALINE_PREP' in t_clean:
        return 'saline_prep'
    elif 'AIR' in t_clean:
        return 'air'
    else:
        print(f"  WARNING: unmapped type '{t_str}' -> 'unknown'")
        return 'unknown'


for case in cases:
    print(f"\n{'='*70}")
    print(f"Processing: {case['name']}")
    print(f"{'='*70}")

    # Load parquet
    df = pd.read_parquet(case['parquet'])
    print(f"Parquet shape: {df.shape}")
    print(f"Time(s) range: {df['Time(s)'].min():.2f} to {df['Time(s)'].max():.2f}")

    # Load annotations
    ann = pd.read_csv(case['csv'])
    print(f"Annotations: {len(ann)} rows")

    # Parse events
    events = []
    for _, row in ann.iterrows():
        t = row['type']
        start = row['start']
        if pd.isna(start) or pd.isna(t):
            continue
        label = classify_label(str(t))
        events.append({'time': float(start), 'label': label})

    events = pd.DataFrame(events).sort_values('time').reset_index(drop=True)
    print(f"Parsed {len(events)} events")
    print(f"Labels: {events['label'].value_counts().to_dict()}")

    # Find classifier window
    start_time = events[events.label == 'start_marker']['time'].iloc[0]
    end_time = events[events.label == 'end_marker']['time'].iloc[0]
    print(f"Classifier window: {start_time:.2f}s to {end_time:.2f}s")

    # Filter to classifier window, exclude markers
    classifier_events = events[(events.time >= start_time) & (events.time <= end_time)]
    classifier_events = classifier_events[~classifier_events.label.isin(['start_marker', 'end_marker'])]
    classifier_events = classifier_events.sort_values('time').reset_index(drop=True)
    print(f"Classifier events: {len(classifier_events)}")

    # Assign labels
    time_col = df['Time(s)'].values
    manual_gt = np.zeros(len(df), dtype=np.uint8)  # 0 = OFF (outside window)

    for i in range(len(classifier_events)):
        t_start = classifier_events.iloc[i]['time']
        t_end = classifier_events.iloc[i + 1]['time'] if i + 1 < len(classifier_events) else end_time
        label_str = classifier_events.iloc[i]['label']
        label_num = label_to_num.get(label_str, 2)  # default to 2 (unknown)

        mask = (time_col >= t_start) & (time_col < t_end)
        manual_gt[mask] = label_num

    df['Manual_GT'] = manual_gt

    # Summary
    print(f"\nManual_GT distribution:")
    vc = pd.Series(manual_gt).value_counts().sort_index()
    code_names = {0: 'OFF', 2: 'UNKNOWN', 4: 'BLOOD', 5: 'CLOT', 6: 'AIR', 8: 'CONTRAST', 9: 'WALL'}
    for code, count in vc.items():
        print(f"  {code} ({code_names.get(code, '?'):>8}): {count:>10,}")

    # Save
    df.to_parquet(case['parquet'], index=False)
    print(f"\nSaved: {case['parquet']}")
