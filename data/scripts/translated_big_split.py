from mt2magic.dataset_utils import get_df_from_txt

import pandas as pd
from sklearn.model_selection import train_test_split

"""
Set the parameters to split cleaned translated datasets in train-dev-test.
"""


seed = 42
src_languages = ('it', 'it') # language of the original sentences
trg_languages = ('en', 'es') # language of the translated sentences
test_size = 1012 # same dimension as the flores dataset
train_size = 0.05 # percentage for validation set
for src_lan, trg_lan in zip(src_languages, trg_languages):
    src_file_path = f"./data/processed/translated/dataset-{src_lan}-{trg_lan}-big/shuf.{src_lan}"
    trg_file_path = f"./data/processed/translated/dataset-{src_lan}-{trg_lan}-big/shuf.{trg_lan}"

    dataset_path = f"./data/processed/translated/translated-big-{src_lan}-{trg_lan}.csv"

    translated_big_df = get_df_from_txt(src_path=src_file_path, trg_path=trg_file_path)
    translated_big_df.to_csv(f"./data/processed/translated/translated-big-{src_lan}-{trg_lan}.csv")

    train_dev, test = train_test_split(translated_big_df, test_size=test_size,
                                       random_state=seed)
    train, dev = train_test_split(train_dev, test_size=train_size,
                                  random_state=seed)

    train.to_csv(f"./data/processed/translated/translated-big-{src_lan}-{trg_lan}-train.csv")
    dev.to_csv(f"./data/processed/translated/translated-big-{src_lan}-{trg_lan}-dev.csv")
    test.to_csv(f"./data/processed/translated/translated-big-{src_lan}-{trg_lan}-test.csv")





