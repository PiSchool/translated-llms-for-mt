from formatting import TranslationData


def get_eval_sentences(src_lan: str, trg_lan: str, num_samples: int=0) -> TranslationData:
    output = {'source': [], 'target': []}
    with open(src_lan) as f:
        examples = f.readlines()
        if num_samples != 0:
            output['source'] = examples[num_samples]
        else:
            output['source'] = examples
    with open(trg_lan) as f:
        examples = f.readlines()
        if num_samples != 0:
            output['target'] = examples[num_samples]
        else:
            output['target'] = examples
    output['translation'] = []
    return output