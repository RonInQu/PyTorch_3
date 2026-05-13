"""
setup_data.py — Copy labeled parquets into training_data/ and test_data/.

Source: LabelingWithDuration.py output in vApril10_data_set/
  - training/ : 7s duration filter (short events blanked to blood)
  - testing/  : full duration (no filter — honest evaluation)

8 fixed test studies → test_data/ (from testing/ source, full duration)
All remaining studies → training_data/ (from training/ source, 7s filtered)
"""
# import os
import shutil
from pathlib import Path

# ── Source directories ──
SRC_TRAIN_DIR = Path('vApril10_data_set') / 'training'
SRC_TEST_DIR  = Path('vApril10_data_set') / 'testing'

# ── Output directories ──
TRAIN_OUT = Path('training_data')
TEST_OUT  = Path('test_data')

# ── Fixed 8 test studies ──
TEST_STUDIES = sorted([
    '33CFB812', '819421BC', '847A1E3F', '8ECEADA6',
    'CENT0008', 'DD2DFAF4', 'F427536B', 'SUMM0127'
])


def get_study_id(filename):
    return filename.split('_labeled_segment')[0]


def main():
    # Clean output dirs (remove parquets only — avoids PermissionError from locks)
    for d in [TRAIN_OUT, TEST_OUT]:
        d.mkdir(exist_ok=True)
        for old in d.glob('*.parquet'):
            old.unlink()

    test_set = set(TEST_STUDIES)

    # Copy training files (7s filtered, excluding test studies)
    n_train = 0
    for f in sorted(SRC_TRAIN_DIR.glob('*_labeled_segment.parquet')):
        study_id = get_study_id(f.name)
        if study_id not in test_set:
            shutil.copy2(f, TRAIN_OUT / f.name)
            n_train += 1

    # Copy test files (full duration)
    n_test = 0
    for study_id in TEST_STUDIES:
        fname = f'{study_id}_labeled_segment.parquet'
        src = SRC_TEST_DIR / fname
        if src.exists():
            shutil.copy2(src, TEST_OUT / fname)
            n_test += 1
        else:
            print(f"WARNING: {study_id} not found in {SRC_TEST_DIR}")

    print(f"Done: {n_train} training (7s filter), {n_test} test (full duration)")


if __name__ == '__main__':
    main()
