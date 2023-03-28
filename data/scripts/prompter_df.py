import pandas as pd
from mt2magic.dataset_utils import get_pool_from_txt
"""
Set src langauge and trg language to build a prompter_df ("pool" value for PrompterConfig)
"""

src_lan = "it"
trg_lan = "es"
mapping_to_flores = {'it': 'ita_Latn', 'en': 'eng_Latn', 'es': 'spa_Latn'}

# set these paths with flores dev-split for both the source and target language
flores_src_path = f"./data/external/flores200_dataset/dev/{mapping_to_flores[src_lan]}.dev"
flores_trg_path =  f"./data/external/flores200_dataset/dev/{mapping_to_flores[trg_lan]}.dev"

# set this path with preprocessed translated dataset, with labelled domain
translated_df_path = f"./data/processed/translated-{src_lan}-{trg_lan}-dev.csv"

# get a pool of examples from the flores dev split
flores_src = get_pool_from_txt(flores_src_path)
flores_trg = get_pool_from_txt(flores_trg_path)
flores_data = {'source': flores_src, 'target': flores_trg}
flores_df = pd.DataFrame(flores_data)
flores_df.to_csv(f"./data/processed/flores-{src_lan}-{trg_lan}-prompt.csv")

# reformat translated data to match mt2magic.formatting.PrompterData
translated_df = pd.read_csv(translated_df_path)
translated_src = translated_df["original"].tolist()
translated_trg = translated_df["translation"].tolist()
translated_lab = translated_df["subject"].tolist()
translated_data = {"source": translated_src, "target": translated_trg,
                   "label": translated_lab}
processed_translated_df = pd.DataFrame(translated_data)
processed_translated_df.to_csv(f"./data/processed/translated-{src_lan}-{trg_lan}-prompt.csv")
