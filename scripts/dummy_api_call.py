from your_package_name.prompting import get_random_prompt
from your_package_name.dataset_utils import get_eval_sentences
import json
import requests
import pandas as pd

"""
Dummy API call to get translations from a LM. 
output csv file in data/interim. WIP
"""

API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

src_ex = "data/external/flores200_dataset/dev/eng_Latn.dev"
trg_ex = "data/external/flores200_dataset/dev/ita_Latn.dev"
src_eval = "data/external/flores200_dataset/devtest/eng_Latn.devtest"
trg_eval = "data/external/flores200_dataset/devtest/ita_Latn.devtest"


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


# this method may be not necessary anymore if the model changes
def extract_string_difference(big_str: str, small_str: str) -> str:
    small_str_len = len(small_str)
    return big_str[small_str_len:]


trans_data = get_eval_sentences(src_lan=src_eval, trg_lan=trg_eval)
trans_data['source'] = trans_data['source'][:10]
trans_data['target'] = trans_data['target'][:10]
# dumb implementation, fix later: we can ask for more than 1 translation for call, depending on the API
for src_sent, trg_sen in zip(trans_data['source'], trans_data['target']):
    prompt = get_random_prompt(src_lan=src_ex, trg_lan=trg_ex, src=src_sent, n=1)
    generated_text = query({'inputs': prompt,  'parameters': {'return_full_text': False,
                            'max_new_tokens': 70} })[0]['generated_text']
    trans_sent = extract_string_difference(generated_text, prompt)
    trans_data['translation'].append(trans_sent)
pass
trans_df = pd.DataFrame.from_dict(trans_data)
trans_df.to_csv(path_or_buf='data/interim/eval_dataframe.csv')
