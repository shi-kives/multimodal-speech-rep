from src.helpers.mapping import labels_mapping
import pandas as pd
from collections import Counter

def sanity_check(embeddings, labels, uids, manifest_path, mode, modality):
    print(f"---- sanity check for {modality} {mode} ----".upper())

    print(f"number of samples in {mode}: {len(labels)}")

    print(f"number of unique utterance ids in {mode}: {uids.nunique()}")

    print(f"shape of train embeddings: {embeddings.shape}")

    print(f"labels mapping: {labels_mapping}")

    df = pd.read_csv(manifest_path)
    emo = dict(Counter(df['Label']).most_common())

    print(f"{mode} set emotions distribution: {emo}")
