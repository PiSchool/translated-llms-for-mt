from random import sample

"""
Methods to perform prompting. WIP
"""


def get_random_prompt(src_lan: str, trg_lan: str, src: str, n: int = 5) -> str:
    with open(src_lan) as f:
        examples = f.readlines()
        size = len(examples)
        idxs = sample(list(range(size)), n)
        src_examples = [examples[idx] for idx in idxs]
    with open(trg_lan) as f:
        examples = f.readlines()
        trans_examples = [examples[idx] for idx in idxs]
    prompt_sentences = [f"[source]: {src_examples[idx]} [target]: {trans_examples[idx]}"
                        for idx in range(n)]
    output = ""
    src_formatting = f"[source]: {src} [target]:"
    for sent in prompt_sentences:
        output += sent
    return output + src_formatting
