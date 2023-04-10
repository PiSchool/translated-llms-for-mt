import pandas as pd
from typing import List


def get_df_from_txt(src_path: str, trg_path: str, src_encoding='utf-8', trg_encoding='utf-8') -> pd.DataFrame:
    output = {'source': [], 'target': []}
    with open(src_path, encoding=src_encoding) as f:
        examples = f.readlines()
        output['source'] = examples
    with open(trg_path, encoding=trg_encoding) as f:
        examples = f.readlines()
        output['target'] = examples
    return pd.DataFrame(data=output)


def get_pool_from_txt(pool_path: str, encoding='utf-8') -> List[str]:
    with open(pool_path, encoding=encoding) as f:
        examples = f.readlines()
        return examples