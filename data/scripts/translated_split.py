import pandas as pd
from sklearn.model_selection import train_test_split

"""
Set the parameters to split cleaned translated datasets in train-dev-test.
"""


seed = 42
src_languages = ('it', 'it') # language of the original sentences
trg_languages = ('en', 'es') # language of the translated sentences
test_size = 1012 # same dimension as the flores dataset
train_size = 0.5 # split dev (for prompting) and train (for fine-tuning) in equal size
for src_lan, trg_lan in zip(src_languages, trg_languages):
    dataset_path = f"./data/processed/translated/translated-{src_lan}-{trg_lan}-cleaned.csv"


    df_translated = pd.read_csv(dataset_path)
    train_dev, test = train_test_split(df_translated, test_size=test_size,
                                       random_state=seed, stratify=df_translated["subject"])
    train, dev = train_test_split(train_dev, test_size=train_size,
                                  random_state=seed, stratify=train_dev["subject"])

    rm_cols = ['src_lang', 'tgt_lang', 'match_type', 'Unnamed: 0']
    train.drop(columns=rm_cols, inplace=True)
    dev.drop(columns=rm_cols, inplace=True)
    test.drop(columns=rm_cols, inplace=True)

    rename_cols = {'original': 'source', 'translation': 'target', 'subject': 'label'}
    train.rename(columns=rename_cols, inplace=True)
    dev.rename(columns=rename_cols, inplace=True)
    test.rename(columns=rename_cols, inplace=True)

    train.to_csv(f"./data/processed/translated/translated-{src_lan}-{trg_lan}-train.csv")
    dev.to_csv(f"./data/processed/translated/translated-{src_lan}-{trg_lan}-dev.csv")
    test.to_csv(f"./data/processed/translated/translated-{src_lan}-{trg_lan}-test.csv")





