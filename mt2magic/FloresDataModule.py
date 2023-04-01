from mt2magic.PEFTDataset import PEFTDataset

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

"""
Class to preprocess and load the Flores dataset before fine-tunings.
Args:
  src_lang (str) : language from which we are translating
  trg_lang (str) : language to which we are translating
  path (str) : path to the Flores dataset
  tokenizer (str) : name of the model we are using
  max_length (int) : parameter used in the tokenizer to control le length of the padding and truncation
  batch_size (int) : dimension of the batch size
  num_workers (int) : if >1 that multi-process data loading is turned on
  prefix (str) : prefix added before the sentence that has to be translated ("Translate from language1 to language2:")
"""
class FloresDataModule(LightningDataModule):
  def __init__(self, 
              src_lang:str, 
              trg_lang:str, 
              path:str, 
              tokenizer:str, 
              max_length:int=128, 
              batch_size:int=32, 
              num_workers:int=1,
              prefix:str="Translate from Italian to Spanish:"
              ):
    super().__init__()

    self.src_lang = src_lang
    self.trg_lang = trg_lang
    self.path = path
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    self.prefix = prefix

  """
  This method transform creates two pd.DataFrame from the files containing the sentences in two different languages
  Args:
    split (str) : use "dev" to create pd.DataFrame from the dev data, "devtest" to do the same from devtest data
  
  Returns:
    :obj:`pd.DataFrame`: df with two columns, one for the source sentences and the second for the target sentences
  """
  def prepare_split(self, split:str="dev"):
    src_sentences = []
    self.src_file = self.path + "{}/{}_Latn.{}".format(split, self.src_lang, split)
    with open(self.src_file, 'r') as f:
      for line in f:
        src_sentences.append(line.strip())
    trg_sentences = []
    self.trg_file = self.path + "{}/{}_Latn.{}".format(split, self.trg_lang, split)
    with open(self.trg_file, 'r') as f:
      for line in f:
        trg_sentences.append(line.strip())
    
    df = pd.DataFrame(list(zip(src_sentences, trg_sentences)), columns=['original', 'translation'])
    return df

  def setup(self, stage:str=None):
    train_data, val_data = train_test_split(self.prepare_split("dev"), test_size=0.2, random_state=42)
    test_data = self.prepare_split("devtest")

    self.X_train_enc, self.X_train_attention, self.Y_train_enc = self.preprocess_data(train_data)
    self.X_val_enc, self.X_val_attention, self.Y_val_enc = self.preprocess_data(val_data)
    self.X_test_enc, self.X_test_attention, self.Y_test_enc = self.preprocess_data(test_data)

  def train_dataloader(self):
    train_dataset = PEFTDataset(self.X_train_enc,self.X_train_attention, self.Y_train_enc)
    return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

  def val_dataloader(self):
    val_dataset = PEFTDataset(self.X_val_enc,self.X_val_attention, self.Y_val_enc)
    return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

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
            [self.prefix+row["original"]], max_length=self.max_length, pad_to_max_length=True, truncation=True
        )
      trg_encoding = self.tokenizer.batch_encode_plus(
            [row["translation"]], max_length=self.max_length, pad_to_max_length=True, truncation=True
        )
      
      input_ids.append(src_encoding.get('input_ids')[0])
      attention_masks.append(src_encoding.get('attention_mask')[0])
      trg_input_ids.append(trg_encoding.get('input_ids')[0])
    
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    trg_input_ids = torch.tensor(trg_input_ids)
    
    return input_ids, attention_masks, trg_input_ids