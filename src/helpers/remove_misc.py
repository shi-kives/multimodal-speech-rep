import csv
import os
from pathlib import Path

csv_path = 'results/artifacts/train_manifest.csv'
col_name = 'Path_to_MP4'
target_dir = Path('data/raw/MELD-RAW/MELD.Raw/train/train_splits')

files_initially = sum(1 for item in target_dir.iterdir() if item.is_file())

keep = set()
with open(csv_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row[col_name]:
            file_path = os.path.normpath(row[col_name].strip())
            keep.add(file_path)

print(list(keep)[:10])

for root, dirs, files in os.walk(target_dir):
    for file in files:
        full_path = os.path.normpath(os.path.join(root, file))

        if full_path not in keep:
            print("deleting: ", full_path)
            os.remove(full_path)

files_left = sum(1 for item in target_dir.iterdir() if item.is_file())
print(f"files initially: {files_initially}, files left: {files_left}\nfiles deleted: {files_initially - files_left}")