import argparse
import pandas as pd
from mt2magic.gpt3_translator import gpt3_translator
from mt2magic.modernMT_translator import modernMT_translator
from mt2magic.t2t_translator import t2tTranslator
from mt2magic.dataset_utils import get_df_from_txt
from mt2magic.evaluator import Evaluator

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
OPEN_AI_API_KEY = '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'
MODERN_MT_KEY = "0AEE783F-6ABF-C594-2CA5-A2DA60708066"
API_URL_T5 = "https://api-inference.huggingface.co/models/t5-base"

src_test_path = f'data/external/flores200_dataset/devtest/{mapping_to_flores[args.src]}.devtest'
trg_test_path = f'data/external/flores200_dataset/devtest/{mapping_to_flores[args.trg]}.devtest'
embeddings_path = f'data/processed/flores-prompt-embedding_{args.src}.pt'
prompt_df = pd.read_csv(f'data/processed/translated/flores-{args.src}-{args.trg}-prompt.csv')

evaluator = Evaluator()
#save_path = f'./data/processed/metrics/{args.model}-3examples-random-{args.src}-{args.trg}-flores.csv'
#aggregate_path = f'./data/processed/metrics/{args.model}-3examples-random-{args.src}-{args.trg}-flores-aggregate.csv'
save_path = f'./data/processed/metrics/{args.model}-{args.src}-{args.trg}-flores.csv'
aggregate_path = f'./data/processed/metrics/{args.model}-{args.src}-{args.trg}-flores-aggregate.csv'
test_df = get_df_from_txt(src_path=src_test_path, trg_path=trg_test_path).head(128)

if args.model == "gpt":
    stop_seq = ['[target]', '[source]']
    param = {'temperature': 0.0, 'max_tokens': 256,'stop': stop_seq}
    prompt_config = {'n_shots': 3, 'strategy': 'random',
                     'pool': prompt_df, 'embeddings_path': embeddings_path}
    gpt3_translator = gpt3_translator(API_KEY=OPEN_AI_API_KEY, prompt_config=prompt_config,
                                     model_name='davinci', param=param)
    gpt3_translator.translate(data=test_df, prompter_config=prompt_config)

if args.model == "mt":
    mt_translator = modernMT_translator(api_key=MODERN_MT_KEY,
                                        source_lang=args.src, target_lang=args.trg)
    mt_translator.translate(data=test_df)

if args.model == "t2t":
    t5_translator = t2tTranslator(API_TOKEN=API_TOKEN, API_URL=API_URL_T5)
    t5_translator.translate(data = test_df, src_lan="Italian", trg_lan="English")


evaluator.evaluating_from_dataframe(dataframe=test_df, save_path=save_path)
corpus_chrf = evaluator.calculate_corpus_chrf(df_evaluation=test_df)
corpus_BLEU = evaluator.calculate_corpus_bleu(df_evaluation=test_df)
aggregate_metrics = {'corpus_chrf': [corpus_chrf], 'corpus_BLEU': [corpus_BLEU]}
metrics_df = pd.DataFrame(aggregate_metrics)
metrics_df.to_csv(aggregate_path)



