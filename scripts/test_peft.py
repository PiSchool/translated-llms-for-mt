from mt2magic.evaluator import Evaluator
from mt2magic.dataset_utils import get_df_from_txt

import argparse
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, AutoTokenizer
import pandas as pd
from tqdm import tqdm

AVAIL_GPUS = 0
if torch.cuda.is_available():       
    device = torch.device("cuda")
    AVAIL_GPUS = torch.cuda.device_count()
    print(f'There are {AVAIL_GPUS} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu") 

def get_predictions(model, df_samples:pd.DataFrame=None, samples_path:str=None, prefix:str="Translate from Italian to Spanish:"):
  if df_samples is not None:
    data = df_samples
  else:
    data = pd.read_csv(samples_path)
  results = []
  for _,s in tqdm(data.iterrows(), total=data.shape[0]):
    message = prefix + s["original"]
    inputs = tokenizer.encode(message, return_tensors="pt", padding=True).to(device)
    output = model.generate(inputs=inputs, max_length=512)
    results.append([s["original"], s["translation"], tokenizer.decode(output[0], skip_special_tokens =True)])

  df = pd.DataFrame(results, columns=["source","target","translation"])
  return df

prefix = "Translate from Italian to Spanish:"

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--dataset')
args = parser.parse_args()

if args.model == "t5":
  model_name = "google/flan-t5-small"
  #model_name = "google/mt5-small"
  #model_name = "google/flan-ul2"
  tokenizer = T5Tokenizer.from_pretrained(model_name)
elif args.model == "bloom":
  model_name = "bigscience/mt0-small"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
   print("Input for model not accepted, possible inputs are: 't5' or 'bloom'")

peft_model_id = "models/LORA_t5"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, peft_model_id)

if args.dataset == "flores":
  src_path = 'data/external/flores200_dataset/devtest/ita_Latn.devtest'
  trg_path = 'data/external/flores200_dataset/devtest/eng_Latn.devtest'
  data_df = get_df_from_txt(src_path=src_path, trg_path=trg_path).head(3)
  data_df.rename(columns = {"source":"original", "target": "translation"}, inplace=True)
  df = get_predictions(model, df_samples=data_df, prefix=prefix)
elif args.dataset == "translated":
  test_path = "data/processed/translated_it-es-test.csv"
  df = get_predictions(model, test_path)

eval = Evaluator()
df_translation = eval.evaluating_from_dataframe(df)

corpus_bleu = eval.calculate_corpus_bleu(df_translation)
mean_bleu = eval.calculate_mean_bleu(df_translation)
corpus_chrf = eval.calculate_corpus_chrf(df_translation)
mean_chrf = eval.calculate_mean_chrf(df_translation)
mean_comet = eval.calculate_system_score_COMET(df_translation)
print('*** *** ***')
print(f'Corpus BLEU: {corpus_bleu}')
print(f'Mean BLEU: {mean_bleu}')
print('*** *** ***')
print(f'Corpus chrf: {corpus_chrf}')
print(f'Mean chrf: {mean_chrf}')
print('*** *** ***')
print(f'\nMean COMET: {mean_comet}')
print('*** *** ***')
