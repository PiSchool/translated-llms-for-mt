import json
import requests
from random import sample

API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
src = "translated-llms-for-mt/data/external/flores200_dataset/dev/eng_Latn.dev"
trans = "translated-llms-for-mt/data/external/flores200_dataset/dev/ita_Latn.dev"


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def get_prompt(src_lan: str, trans_lan: str, src: str, n: int = 5) -> str:
    with open(src_lan) as f:
        examples = f.readlines()
        size = len(examples)
        idxs = sample(list(range(size)), n)
        src_examples = [examples[idx] for idx in idxs]
    with open(trans_lan) as f:
        examples = f.readlines()
        trans_examples = [examples[idx] for idx in idxs]
    prompt_sentences = [f"[source]: {src_examples[idx]} [target]: {trans_examples[idx]}"
                        for idx in range(n)]
    output = ""
    src_formatting = f"[source]: {src} [target]:"
    for sent in prompt_sentences:
        output += sent
    return output + src_formatting


src_sent = "Il cane cammina nel parco.\n"
prompt = get_prompt(src_lan=src, trans_lan=trans, src=src_sent, n=1)
data = query({'inputs': prompt, 'temperature': 0.5, 'top_k': 2, 'return_full_text': False})
pass
