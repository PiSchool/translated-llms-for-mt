import argparse
import pandas as pd
from mt2magic.gpt_translator import gptTranslator
from mt2magic.mt_translator import mtTranslator
from mt2magic.t2t_translator import t2tTranslator
from mt2magic.dataset_utils import get_df_from_txt
# from scripts.Evaluator import Evaluator

"""
For now src can be only it, trg can be either es or en.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--src')
parser.add_argument('--trg')
args = parser.parse_args()
mapping_to_flores = {'it': 'ita_Latn', 'en': 'eng_Latn', 'es': 'spa_Latn'}

API_TOKEN = "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"
API_URL_GPT = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
API_URL_T5 = "https://api-inference.huggingface.co/models/t5-base"

src_test_path = f'data/external/flores200_dataset/devtest/{mapping_to_flores[args.src]}.devtest'
trg_test_path = f'data/external/flores200_dataset/devtest/{mapping_to_flores[args.trg]}.devtest'
embeddings_path = f'data/processed/flores-prompt-embedding_{args.src}.pt'
prompt_df = pd.read_csv(f'data/processed/translated-{args.src}-{args.trg}-prompt.csv')

# evaluator = Evaluator()
save_path = './data/processed/metrics/check.csv'
test_df = get_df_from_txt(src_path=src_test_path, trg_path=trg_test_path).head(5)

if args.model == "gpt":
    param = {'return_full_text': False, 'wait_for_model': True, 'max_new_tokens': 80}
    prompt_config = {'n_shots': 3, 'strategy': 'random',
                     'pool': prompt_df, 'embeddings_path': embeddings_path}
    gpt_translator = gptTranslator(API_TOKEN=API_TOKEN, API_URL=API_URL_GPT,
                               prompt_config=prompt_config, parameters=param)
    gpt_translator.translate(data=test_df, prompter_config=prompt_config)

if args.model == "mt":
    mt_translator = mtTranslator(API_TOKEN=API_TOKEN, API_URL="check.csv")
    mt_translator.translate(data=test_df, src_lan='it', trg_lan='en')

if args.model == "t2t":
    t5_translator = t2tTranslator(API_TOKEN=API_TOKEN, API_URL=API_URL_T5)
    t5_translator.translate(data = test_df, src_lan="Italian", trg_lan="English")

test_df.to_csv(save_path)

# evaluator.evaluating_from_dataframe(dataframe=translation_df, save_path=save_path)



