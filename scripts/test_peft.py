from Evaluator import Evaluator
from mt2magic.dataset_utils import get_df_from_txt

import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import pandas as pd

def get_predictions(model, samples, target, prefix):
  results = []
  for i,m in enumerate(samples):
    message = prefix + m
    inputs = tokenizer.encode(message, return_tensors="pt")#.to("cuda")
    output = model.generate(inputs=inputs)
    results.append([m, tokenizer.decode(output[0]), target[i]])

  df = pd.DataFrame(results, columns=["source","target","translation"])
  return df

parser = argparse.ArgumentParser()
parser.add_argument('--model_path')
args = parser.parse_args()

model_name = "google/mt5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

peft_model_id = "/home/gianfree/Desktop/translated-llms-for-mt/models/LORA_t5"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

src_path = 'data/external/flores200_dataset/devtest/ita_Latn.devtest'
trg_path = 'data/external/flores200_dataset/devtest/eng_Latn.devtest'

translation_df = get_df_from_txt(src_path=src_path, trg_path=trg_path).head(3)

prefix = "Translate from Italian to Spanish: "
df = get_predictions(model, translation_df["source"], translation_df["target"], prefix)

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
