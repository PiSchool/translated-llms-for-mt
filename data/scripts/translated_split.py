import pandas as pd
from sklearn.model_selection import train_test_split

"""
Set the parameters to split a cleaned translated dataset in train-dev-test.
"""


seed = 42
src_lan = 'it' # language of the original sentences
trg_lan = 'es' # language of the translated sentences
test_size = 1012 # dimension of the flores dataset
train_size = 0.5 # split dev (for prompting) and train (for fine-tuning) in equal size
dataset_path = f"./data/processed/{src_lan}-{trg_lan}-cleaned.csv"


df_translated = pd.read_csv(dataset_path)
train_dev, test = train_test_split(df_translated, test_size=test_size,
                                   random_state=seed, stratify=df_translated["subject"])
train, dev = train_test_split(train_dev, test_size=train_size,
                              random_state=seed, stratify=train_dev["subject"])

train.to_csv(f"./data/processed/translated-{src_lan}-{trg_lan}-train.csv")
dev.to_csv(f"./data/processed/translated-{src_lan}-{trg_lan}-dev.csv")
test.to_csv(f"./data/processed/translated-{src_lan}-{trg_lan}-test.csv")




