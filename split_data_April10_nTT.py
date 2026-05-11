from pathlib import Path
import shutil

BASE_DIR = Path(r"C:\Users\RonaldKurnik\OneDrive - Inquis Medical\Documents\2026\PyTorch_3")

SOURCE_TEST  = BASE_DIR / "vApril10_data_set" / "testing"
SOURCE_TRAIN = BASE_DIR / "vApril10_data_set" / "training"

TARGET_TEST  = BASE_DIR / "test_data"
TARGET_TRAIN = BASE_DIR / "training_data"

TARGET_TEST.mkdir(exist_ok=True)
TARGET_TRAIN.mkdir(exist_ok=True)

TEST_PREFIXES = {
    '00F628C9', '33CFB812', '43140EA7', '4E3747A0', '530618CC', '819421BC',
    'A225B105', 'AFF18ECE', 'CENT0008', 'HACK0140', 'HUNT0120', 'STCLD001',
    'SUMM0127', 'UHMAX001'
}

TRAIN_PREFIXES = {
    '05EA15A5', '09419CF3', '0D9C36A0', '15A93526', '15AC6217', '16621B3E',
    '18A9A741', '1A8F0795', '24AFD80C', '26E955BA', '325A317A', '34268034',
    '376DCB0D', '3B90D74B', '3E146478', '42CF0AE3', '453F37DC', '4633BDC0',
    '48663E05', '4B4BF4DB', '50ACAF6E', '58F78079', '5A31F836', '6E7EB56C',
    '71119917', '73CB9CA1', '743CBF58', '7873BF1D', '81FC0C79', '847A1E3F',
    '86FA6755', '8860D580', '8ECEADA6', '8EE40C79', '903FE519', '9C63125D',
    'B58B74D7', 'B9E8EB7F', 'BAPT0001', 'CENT0006', 'CENT0007', 'CENT0009',
    'CENT0102', 'CENT0161', 'CENT0165', 'CENT0176', 'CENT0182', 'CENT0231',
    'D25DD102', 'D4793E80', 'DBEF90C4', 'DD2DFAF4', 'EA7C0500', 'ELCA0179',
    'F39B2DEA', 'F427536B', 'F60DF902', 'FE454F2D', 'HUNT0130', 'HUNT0134',
    'HUNT0136', 'HUNT0150', 'HUNT0159', 'HUNT0177', 'HUNT0178', 'HUNT0198',
    'NASHUN01', 'NASHUN02', 'SOMI0153', 'STCL0090', 'STCLD002', 'SUMM0119',
    'SUMM0149', 'SUMM0152', 'SUMM0154', 'SUMM0163', 'SUMM0183', 'UH000008',
    'UNIH0148'
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