import whisper
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json

model = whisper.load_model("base")

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[2]

manifest_path = ROOT / "results" / "artifacts" / "train_manifest.csv"
manifest = pd.read_csv(manifest_path)
transcripts = []

for row in tqdm(manifest.itertuples(index=True), total = len(manifest), desc = "extracting whisper transcripts"):

    abs_path = ROOT / Path(row.Path_to_WAV)

    if Path(abs_path).is_file():
        result = model.transcribe(str(abs_path))
        transcripts.append({
            'SNo': row.SNo,
            'Whisper_Transcript': result["text"],
            'Dialogue_ID': row.Dialogue_ID, 
            'Utterance_ID': row.Utterance_ID, 
            'Speaker': row.Speaker,
            'Label': row.Label
        })

    else:
        print(f"error! couldn't find file: dia{row.Dialogue_ID}_utt{row.Utterance_ID}")
        continue

transcripts_path = Path(ROOT / "results" / "artifacts" / "whisper_transcripts.json")
transcripts_path.parent.mkdir(parents=True, exist_ok=True)

with open(transcripts_path, "w", encoding='utf-8') as f:
    json.dump(transcripts, f, ensure_ascii = False, indent = 4)

print("Whisper generated transcripts saved at: ", transcripts_path)