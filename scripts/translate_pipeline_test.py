import argparse
from mt2magic.gpt_translator import gptTranslator
from mt2magic.mt_translator import mtTranslator
from mt2magic.t2t_translator import t2tTranslator
from mt2magic.dataset_utils import get_df_from_txt, get_pool_from_txt

parser = argparse.ArgumentParser()
parser.add_argument('--model')
args = parser.parse_args()

API_TOKEN = "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"
API_URL_GPT = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
API_URL_T5 = "https://api-inference.huggingface.co/models/t5-base"

src_path = 'data/external/flores200_dataset/devtest/ita_Latn.devtest'
trg_path = 'data/external/flores200_dataset/devtest/eng_Latn.devtest'

src_pool_path = 'data/external/flores200_dataset/dev/ita_Latn.dev'
trg_pool_path = 'data/external/flores200_dataset/dev/eng_Latn.dev'

translation_df = get_df_from_txt(src_path=src_path, trg_path=trg_path).head(3)
src_pool = get_pool_from_txt(src_pool_path)
trg_pool = get_pool_from_txt(trg_pool_path)

if args.model == "gpt":
    param = {'return_full_text': False, 'wait_for_model': True, 'max_new_tokens': 80}
    prompt_config = {'n_shots': 2, 'strategy': "random", "src_pool": src_pool, "trg_pool": trg_pool, "model": None}
    gpt_translator = gptTranslator(API_TOKEN=API_TOKEN, API_URL=API_URL_GPT,
                               prompt_config=prompt_config, parameters=param)
    gpt_translator.translate(data=translation_df, prompter_config=prompt_config)

if args.model == "mt":
    mt_translator = mtTranslator(API_TOKEN=API_TOKEN, API_URL="check")
    mt_translator.translate(data=translation_df, src_lan='it', trg_lan='en')

if args.model == "t2t":
    t5_translator = t2tTranslator(API_TOKEN=API_TOKEN, API_URL=API_URL_T5)
    t5_translator.translate(data = translation_df, src_lan="Italian", trg_lan="English")



