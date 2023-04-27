from mt2magic.utils.prompter import Prompter

from torch.utils.data import Dataset
import pandas as pd

"""
Class to preprocess and load the test set from the Translated dataset
Args:
  test_file (str) : path to the csv that contains the test data
  test_file (str) : path to the csv that contains the test data
  prompt_type(str) : determine the class of prompt used in the tests. More info in mt2magic/prompter_bloom.py
  src_lan (str) : language of the source text
  trg_lan (str) : language of the target text
"""
class TestPEFTDataset(Dataset):
    def __init__(self, test_path:str, prompt_type:str, src_lan:str, trg_lan:str):
        self.test_data = pd.read_csv(test_path)
        self.prompt_type = prompt_type
        self.src_lan = src_lan
        self.trg_lan = trg_lan

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        text = Prompter.BLOOM_prompt(type=self.prompt_type,
                            sentence=self.test_data["source"].iloc[idx], 
                            src_lan=self.src_lan, 
                            trg_lan=self.trg_lan
                            )
        return text