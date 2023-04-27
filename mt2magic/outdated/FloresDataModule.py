from mt2magic.peft.PEFTDataset import PEFTDataset

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

"""
Class to preprocess and load the Flores dataset before fine-tunings.
Args:
  dev_path (str) : path to the csv that contains the dev data
  devtest_path (str) : path to the csv that contains the devtest data
  tokenizer (str) : name of the model we are using
  max_length (int) : parameter used in the tokenizer to control le length of the padding and truncation
  batch_size (int) : dimension of the batch size
  num_workers (int) : if >1 that multi-process data loading is turned on
  prefix (str) : prefix added before the sentence that has to be translated ("Translate from language1 to language2:")
"""
class FloresDataModule(LightningDataModule):
  def __init__(self, 
              dev_path:str, 
              devtest_path:str,
              tokenizer:str, 
              max_length:int=128, 
              batch_size:int=32, 
              num_workers:int=1,
              prefix:str="Translate from Italian to Spanish:"
              ):
    super().__init__()

    self.dev_path = dev_path
    self.devtest_path = devtest_path
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    self.prefix = prefix

  def setup(self, stage:str=None):
    train_df = pd.read_csv(self.dev_path)
    test_data = pd.read_csv(self.devtest_path)
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

    self.X_train_enc, self.X_train_attention, self.Y_train_enc = self.preprocess_data(train_data)
    self.X_val_enc, self.X_val_attention, self.Y_val_enc = self.preprocess_data(val_data)
    self.X_test_enc, self.X_test_attention, self.Y_test_enc = self.preprocess_data(test_data)

  def train_dataloader(self):
    train_dataset = PEFTDataset(self.X_train_enc,self.X_train_attention, self.Y_train_enc)
    return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

  def val_dataloader(self):
    val_dataset = PEFTDataset(self.X_val_enc,self.X_val_attention, self.Y_val_enc)
    return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

  def test_dataloader(self):
    test_dataset = PEFTDataset(self.X_test_enc, self.X_test_attention, self.Y_test_enc)
    return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

  """
  This method apply the tokenizer (after the addition of the prefix) to the sentences 
  and  it also obtains the input_ids, attention_mask and labels
  Args:
    data (pd.DataFrame) : split of data we want to use the tokenizer on

  Returns: input_ids, attention_mask and labels
  """
  def preprocess_data(self, data:pd.DataFrame):
    input_ids = []
    attention_masks = []
    trg_input_ids = []
    for _, row in data.iterrows():
      src_encoding = self.tokenizer.batch_encode_plus(
            [self.prefix+row["source"]], max_length=self.max_length,  padding="max_length", truncation=True
        )
      trg_encoding = self.tokenizer.batch_encode_plus(
            [row["target"]], max_length=self.max_length,  padding="max_length", truncation=True
        )
      
      input_ids.append(src_encoding.get('input_ids')[0])
      attention_masks.append(src_encoding.get('attention_mask')[0])
      trg_input_ids.append(trg_encoding.get('input_ids')[0])
    
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    trg_input_ids = torch.tensor(trg_input_ids)
    
    return input_ids, attention_masks, trg_input_ids