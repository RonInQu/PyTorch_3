from pathlib import Path
import shutil

BASE_DIR = Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3")

SOURCE_TEST  = BASE_DIR / "vApril6_data_set" / "testing"
SOURCE_TRAIN = BASE_DIR / "vApril6_data_set" / "training"

TARGET_TEST  = BASE_DIR / "test_data"
TARGET_TRAIN = BASE_DIR / "training_data"

TARGET_TEST.mkdir(exist_ok=True)
TARGET_TRAIN.mkdir(exist_ok=True)

TEST_PREFIXES = {
    '33CFB812', '819421BC', '847A1E3F', '8ECEADA6',
    'CENT0008', 'DD2DFAF4', 'F427536B', 'SUMM0127'
}

TRAIN_PREFIXES = {
    '15AC6217', '16621B3E', '1A8F0795', '376DCB0D', '3B90D74B', '43140EA7', 
    '48663E05', '530618CC', '6E7EB56C', '73CB9CA1', '743CBF58', '7873BF1D',
    '81FC0C79', '8860D580', '9C63125D', 'AFF18ECE', 'BAPT0001', 'CENT0007',
    'CENT0009', 'D25DD102', 'D4793E80', 'DBEF90C4', 'EA7C0500', 'ELCA0179',
    'F39B2DEA', 'FE454F2D', 'HUNT0120', 'HUNT0134', 'HUNT0136', 'NASHUN01',
    'STCLD002', 'SUMM0119', 'SUMM0152', 'UHMAX001'
}

print("Moving TEST files...")
for f in SOURCE_TEST.glob("*.parquet"):
    prefix = f.stem.replace('_labeled_segment', '')
    if prefix in TEST_PREFIXES:
        shutil.move(str(f), str(TARGET_TEST / f.name))
        print(f"→ test_data : {f.name}")

print("\nMoving TRAINING files...")
for f in SOURCE_TRAIN.glob("*.parquet"):
    prefix = f.stem.replace('_labeled_segment', '')
    if prefix in TRAIN_PREFIXES:
        shutil.move(str(f), str(TARGET_TRAIN / f.name))
        print(f"→ training_data : {f.name}")

print("\n" + "="*60)
print("DONE! Only the specified files were moved.")