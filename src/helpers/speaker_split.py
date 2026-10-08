import pandas as pd

train = pd.read_csv('data/raw/MELD-RAW/MELD.Raw/train_sent_emo.csv')
val = pd.read_csv('data/raw/MELD-RAW/MELD.Raw/dev_sent_emo.csv')
test = pd.read_csv('data/raw/MELD-RAW/MELD.Raw/test_sent_emo.csv')

train_split = ['Phoebe', 'Monica', 'Chandler', 'Janice']
val_split = ['Rachel', 'Ross']
test_split = ['Joey']

train = train[train['Speaker'].isin(train_split)]
val = val[val['Speaker'].isin(val_split)]
test = test[test['Speaker'].isin(test_split)]

if __name__ == '__main__':
    print("split done.\n")
    print("verification: ")
    print("number of speakers in train (should be 4): ", train['Speaker'].nunique())
    print("number of speakers in val (should be 2): ", val['Speaker'].nunique())
    print("number of speakers in test (should be 1): ", test['Speaker'].nunique())

    print("\nnumber of samples after splitting:")
    print("train: ", len(train))
    print("val: ", len(val))
    print("test: ", len(test))