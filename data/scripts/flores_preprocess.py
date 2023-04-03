import pandas as pd
from mt2magic.dataset_utils import get_pool_from_txt
from mt2magic.dataset_utils import get_df_from_txt

"""
Set src langauge and trg language to build a prompter_df ("pool" value for PrompterConfig)
"""

src_lan = "it"
trg_lan = "es"
mapping_to_flores = {'it': 'ita_Latn', 'en': 'eng_Latn', 'es': 'spa_Latn'}

# set these paths with flores dev-split for both the source and target language
dev_src_path = f"./data/external/flores200_dataset/dev/{mapping_to_flores[src_lan]}.dev"
dev_trg_path =  f"./data/external/flores200_dataset/dev/{mapping_to_flores[trg_lan]}.dev"

# get a pool of examples from the flores dev split
dev_src = get_pool_from_txt(dev_src_path)
dev_trg = get_pool_from_txt(dev_trg_path)
dev_data = {'source': dev_src, 'target': dev_trg}
dev_df = pd.DataFrame(dev_data)
dev_df.to_csv(f"./data/processed/flores/flores-{src_lan}-{trg_lan}-dev.csv")

# create a test set for the flores dataset
test_src_path = f"./data/external/flores200_dataset/devtest/{mapping_to_flores[src_lan]}.devtest"
test_trg_path =  f"./data/external/flores200_dataset/devtest/{mapping_to_flores[trg_lan]}.devtest"
test_df = get_df_from_txt(src_path=test_src_path, trg_path=test_trg_path)
test_df.to_csv(f"./data/processed/flores/flores-{src_lan}-{trg_lan}-test.csv")

