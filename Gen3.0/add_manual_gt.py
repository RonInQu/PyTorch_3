"""Add Manual_GT column to Promedica parquet based on manual annotations CSV."""
import pandas as pd
import numpy as np

# Load parquet
parquet_file = '2026-05-12 206-104 Promedica_state.parquet'
df = pd.read_parquet(parquet_file)

print(f"Parquet shape: {df.shape}")
print(f"Time(s) range: {df['Time(s)'].min():.2f} to {df['Time(s)'].max():.2f}")

# Load annotations CSV
csv_path = r"c:\Users\RonaldKurnik\Inquis Medical\DataScience - Documents\Pipeline\Staging_IDE_Powered\2026-05-12 206-104 Promedica\events260-104 Promedica.csv"
ann = pd.read_csv(csv_path)
print(f"\nAnnotations shape: {ann.shape}")
print(f"Columns: {list(ann.columns)}")
print(f"\nUnique types:")
for t in ann['type'].dropna().unique():
    print(f"  '{t}'")

# Map annotation labels to tissue classes
# User specified: Tracking=blood, Wall Latch=wall, Clot=clot
label_map = {
    'TRACKING ': 'blood',
    'TRACKING': 'blood',
    'WALL_LATCH': 'wall',
    'WALL_LATCH\u200b': 'wall',       # with zero-width space
    'WALL_LATCH\u200b\u200b': 'wall', # with double zero-width space
    'CLOT': 'clot',
    'CLOT_IN_AREA ': 'clot',
    'CLOT_IN_AREA': 'clot',
    'CLOT_WALL (unknown while under vacuum)\u200b': 'unknown',
    'CLOT_WALL (unknown while under vacuum)': 'unknown',
    'CONTRAST': 'contrast',
    'UNKNOWN/UNGRADABLE\u200b': 'unknown',
    'UNKNOWN/UNGRADABLE': 'unknown',
    'ENTRY_EXIT': 'entry_exit',
    'START_HERE_FOR_CLASSIFIER \u200b': 'start_marker',
    'START_HERE_FOR_CLASSIFIER': 'start_marker',
    'END_HERE_FOR_CLASSIFIER ': 'end_marker',
    'END_HERE_FOR_CLASSIFIER': 'end_marker',
}

# Clean the type column (strip whitespace variants)
ann['type_clean'] = ann['type'].apply(lambda x: str(x).strip() if pd.notna(x) else '')

# Build the mapping: for each row, get the start time and mapped label
events = []
for _, row in ann.iterrows():
    t = row['type']
    start = row['start']
    if pd.isna(start) or pd.isna(t):
        continue
    # Try to map the label (try with and without trailing spaces/zero-width)
    label = None
    t_str = str(t)
    for key, val in label_map.items():
        if t_str.strip().replace('\u200b', '').replace('\u200c', '') == key.strip().replace('\u200b', '').replace('\u200c', ''):
            label = val
            break
    if label is None:
        # Try partial match
        t_clean = t_str.strip().replace('\u200b', '').replace('\u200c', '')
        if 'TRACKING' in t_clean:
            label = 'blood'
        elif 'WALL_LATCH' in t_clean:
            label = 'wall'
        elif 'CLOT_IN_AREA' in t_clean:
            label = 'clot'
        elif 'CLOT_WALL' in t_clean:
            label = 'unknown'
        elif 'CLOT' in t_clean:
            label = 'clot'
        elif 'CONTRAST' in t_clean:
            label = 'contrast'
        elif 'UNKNOWN' in t_clean or 'UNGRADABLE' in t_clean:
            label = 'unknown'
        elif 'ENTRY_EXIT' in t_clean:
            label = 'entry_exit'
        elif 'START_HERE' in t_clean:
            label = 'start_marker'
        elif 'END_HERE' in t_clean:
            label = 'end_marker'
        else:
            label = 'unknown'
            print(f"  WARNING: unmapped type '{t_str}' -> 'unknown'")

    events.append({'time': float(start), 'label': label})

events = pd.DataFrame(events).sort_values('time').reset_index(drop=True)
print(f"\nParsed {len(events)} events")
print(f"Label distribution:")
print(events['label'].value_counts().to_string())

# Find the classifier window
start_time = events[events.label == 'start_marker']['time'].iloc[0]
end_time = events[events.label == 'end_marker']['time'].iloc[0]
print(f"\nClassifier window: {start_time:.2f}s to {end_time:.2f}s")

# Assign Manual_GT to parquet
# Each annotation label applies from its start time until the next annotation's start time
time_col = df['Time(s)'].values

# Initialize as empty string
manual_gt = np.full(len(df), '', dtype=object)

# Sort events by time and assign labels
# Only use events within the classifier window
classifier_events = events[(events.time >= start_time) & (events.time <= end_time)]
classifier_events = classifier_events[~classifier_events.label.isin(['start_marker', 'end_marker'])]
classifier_events = classifier_events.sort_values('time').reset_index(drop=True)

print(f"\nClassifier events: {len(classifier_events)}")

# For each event, mark all samples from its time until the next event's time
for i in range(len(classifier_events)):
    t_start = classifier_events.iloc[i]['time']
    t_end = classifier_events.iloc[i + 1]['time'] if i + 1 < len(classifier_events) else end_time
    label = classifier_events.iloc[i]['label']

    mask = (time_col >= t_start) & (time_col < t_end)
    manual_gt[mask] = label

# Mark anything outside classifier window as empty
outside = (time_col < start_time) | (time_col >= end_time)
manual_gt[outside] = ''

df['Manual_GT'] = manual_gt

# Summary
print(f"\nManual_GT distribution (non-empty):")
gt_counts = pd.Series(manual_gt[manual_gt != '']).value_counts()
print(gt_counts.to_string())

# Compare with state machine labels
tissue_map = {4: 'blood', 5: 'clot', 9: 'wall'}
df['auto_tissue'] = df['light_style_i'].map(tissue_map).fillna('other')
cross = pd.crosstab(df[df.Manual_GT != '']['Manual_GT'], df[df.Manual_GT != '']['auto_tissue'])
print(f"\nCross-tabulation (Manual_GT rows vs auto_tissue columns):")
print(cross.to_string())

# Save
df.to_parquet(parquet_file, index=False)
print(f"\nSaved updated parquet with Manual_GT column: {parquet_file}")
