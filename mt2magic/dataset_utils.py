import pandas as pd
from typing import List

def get_df_from_txt(src_path: str, trg_path: str) -> pd.DataFrame:
    output = {'source': [], 'target': []}
    with open(src_path) as f:
        examples = f.readlines()
        output['source'] = examples
    with open(trg_path) as f:
        examples = f.readlines()
        output['target'] = examples
    return pd.DataFrame(data=output)

def get_pool_from_txt(pool_path: str) -> List[str]:
    with open(pool_path) as f:
        examples = f.readlines()
        return examples
