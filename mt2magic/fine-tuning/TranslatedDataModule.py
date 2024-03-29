from mt2magic.peft.PEFTDataset import PEFTDataset
from mt2magic.utils.prompter import Prompter
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
import pandas as pd

"""
Class to preprocess and load the Translated dataset before fine-tunings.
Args:
  train_file (str) : path to the csv that contains the train data
  test_file (str) : path to the csv that contains the test data
  val_file (str) : path to the csv that contains the val data
  tokenizer (str) : name of the model we are using
  max_length (int) : parameter used in the tokenizer to control le length of the padding and truncation
  batch_size (int) : dimension of the batch size
  num_workers (int) : if >1 that multi-process data loading is turned on 
  prefix_type (str) : determine the class of prompt used in the tests. More info in mt2magic/prompter_bloom.py
  src_lan (str) : language of the source text
  trg_lan (str) : language of the target text
  limit (int) : limit the dimension of the train set
"""

class TranslatedDataModule(LightningDataModule):
  def __init__(self, 
              train_file:str, 
              test_file:str, 
              val_file:str, 
              tokenizer:str, 
              max_length:int=128, 
              batch_size:int=32, 
              num_workers:int=0,
              prefix_type:str="B",
              src_lan:str="Italian",
              trg_lan:str="English",
              limit:int=0
              ):
    super().__init__()

    self.train_file = train_file
    self.test_file = test_file
    self.val_file = val_file
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    self.prefix_type = prefix_type
    self.src_lan = src_lan
    self.trg_lan = trg_lan
    self.limit = limit

  def setup(self, stage:str=None):
    if self.limit != 0:
      train_data = pd.read_csv(self.train_file).iloc[:self.limit]
      val_data = pd.read_csv(self.val_file).iloc[:self.limit]
    else:
      train_data = pd.read_csv(self.train_file)
      val_data = pd.read_csv(self.val_file)

    #test_data = pd.read_csv(self.test_file)
    
    self.X_train_enc, self.X_train_attention, self.Y_train_enc = self.preprocess_data(train_data)
    self.X_val_enc, self.X_val_attention, self.Y_val_enc = self.preprocess_data(val_data)
    #self.X_test_enc, self.X_test_attention, self.Y_test_enc = self.preprocess_data(test_data)

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
            [Prompter.BLOOM_prompt(type=self.prefix_type, sentence=row["source"], src_lan=self.src_lan, trg_lan=self.trg_lan)], max_length=self.max_length,  padding="max_length", truncation=True
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