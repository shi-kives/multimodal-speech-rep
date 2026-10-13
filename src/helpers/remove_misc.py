import csv
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

csv_path = ROOT / 'results/artifacts/test_manifest.csv'
col_name = 'Path_to_WAV'
target_dir = Path(ROOT / 'data/raw/MELD-RAW/MELD.Raw/test/output_repeated_splits_test/')

files_initially = sum(1 for item in target_dir.iterdir() if item.is_file())

keep = set()
with open(csv_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row[col_name]:
            temp = str(ROOT) + '/' + row[col_name]
            file_path = os.path.normpath(temp.strip())
            keep.add(file_path)

print(list(keep)[:10])
print("KEEP:", len(list(keep)))

for root, dirs, files in os.walk(target_dir):
    for file in files:
        full_path = os.path.normpath(os.path.join(root, file))
        if full_path not in keep:
            print("deleting: ", full_path)
            os.remove(full_path)

files_left = sum(1 for item in target_dir.iterdir() if item.is_file())
print(f"files initially: {files_initially}, files left: {files_left}\nfiles deleted: {files_initially - files_left}")