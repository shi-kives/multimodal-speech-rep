'''combining features: 
utterance id, dialogue id, speaker, emotion (label),
reference transcript (i.e, utterance), path to wav file'''

import pandas as pd
from speaker_split import train, val, test

test['Path_to_WAV'] = test.apply(lambda row: f"data/raw/MELD-RAW/MELD.Raw/test/output_repeated_splits_test/dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav", axis = 1)

manifest = pd.DataFrame(test[['Sr No.', 'Dialogue_ID', 'Utterance_ID', 'Speaker', 'Emotion', 'Utterance', 'Path_to_WAV']])
col_names = ['SNo', 'Dialogue_ID', 'Utterance_ID', 'Speaker', 'Label', 'Reference_Transcript', 'Path_to_WAV']

manifest.to_csv('results/artifacts/test_manifest.csv', header = col_names, index = False)
print("manifest created.")