from typing import TypedDict, List


class TranslationData(TypedDict):
    source: List[str]
    target: List[str]
    translation: List[str]


def get_eval_sentences(src_lan: str, trg_lan: str) -> TranslationData:
    output = {'source': [], 'target': []}
    with open(src_lan) as f:
        examples = f.readlines()
        output['source'] = examples
    with open(trg_lan) as f:
        examples = f.readlines()
        output['target'] = examples
    output['translation'] = []
    return output