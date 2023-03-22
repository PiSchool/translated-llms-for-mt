from torch.utils.data import Dataset

"""
Class to load and prepare the Flores dataset before fine-tuning a model

"""
class FloresDataset(Dataset):
  def __init__(self, src_file, trg_file, tokenizer, prefix, max_length=128):

    self.src_sentences = []
    self.trg_sentences = []
    self.tokenizer = tokenizer
    self.max_length = max_length
    
    with open(src_file, 'r') as f:
      i = 0
      for line in f:
        if i==100:
          break
        self.src_sentences.append(prefix + line.strip())
        i+=1
    
    with open(trg_file, 'r') as f:
      j=0
      for line in f:
        if j == 100:
          break
        self.trg_sentences.append(line.strip())
        j+=1

  def __len__(self):
    return len(self.src_sentences)

  def __getitem__(self, index):
    src_encoding = self.tokenizer(self.src_sentences[index], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
    
    trg_encoding = self.tokenizer(self.trg_sentences[index], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
    
    input_ids = src_encoding['input_ids'].squeeze()
    attention_mask = src_encoding['attention_mask'].squeeze()
    trg_input_ids = trg_encoding['input_ids'].squeeze()
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': trg_input_ids}