'''combining features: 
utterance id, dialogue id, speaker, emotion (label),
reference transcript (i.e, utterance), path to mp4 file'''

import pandas as pd
from speaker_split import train, val, test

train['Path_to_MP4'] = train.apply(lambda row: f"data/raw/MELD-RAW/MELD.Raw/train/train_splits/dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4", axis = 1)

manifest = pd.DataFrame(train[['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Emotion', 'Utterance', 'Path_to_MP4']])
col_names = ['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Label', 'Reference_Transcript', 'Path_to_MP4']

manifest.to_csv('results/artifacts/train_manifest.csv', header = col_names, index = False)
print("manifest created.")