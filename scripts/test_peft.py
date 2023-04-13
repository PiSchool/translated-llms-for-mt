from mt2magic.evaluator import Evaluator
from mt2magic.TestPEFTDataset import TestPEFTDataset

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import pandas as pd
from tqdm import tqdm

from omegaconf import DictConfig
import hydra

"""
  This method generate the translations for the samples in the test set 
  using a fine-tuned model (fine-tuned with LoRA)
  and returns a pandas DataFrame that will be used to compute metrics
  Args:
    lora_module (str) : path to the folder that contains the config file for LoRA
    test_path (str) : path to the csv file that contains the source sentences and the target sentences
    batch_size (int) : dimensione of the batches
    device (str) : device used for the inference (gpu or cpu)
    prefix (str) : prefix to append to the source sentence
    src_lan (str) : language of the source text
    trg_lan (str) : language of the target text
    limit (int) : number of samples of the test set used for evaluation
  Returns: 
    df (pd.DataFrame) : pandas DataFrame with the source sentences, the target sentences, and the translations
"""
def get_predictions(lora_module:str,
                    test_path:str,
                    batch_size:int, 
                    device:str, 
                    prefix:str,
                    src_lan:str,
                    trg_lan:str,
                    limit:int=0
                    ):
    config = PeftConfig.from_pretrained(lora_module)
    # Loading the model in int8 is going to slow down inference for some reason
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)#,  load_in_8bit=True,  device_map='auto')
    model = PeftModel.from_pretrained(model, lora_module).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model.eval()

    translator = pipeline(f"translation_{src_lan}_to_{trg_lan}", model=model, tokenizer=tokenizer, device=device)
    dataset = TestPEFTDataset(test_path, prefix)
    results = []
    df = pd.read_csv(test_path)
    if limit > 0:
        df = df.iloc[:limit]
    elif limit == 0:
        limit = len(df)
    for translated_text in translator(dataset, batch_size=batch_size):
        results.append(translated_text[0]["translation_text"])
        if limit > 1:
            limit -= 1
        elif limit == 1:
            break
    
    df["translation"] = results
    res = df[["source", "target", "translation"]]
    return res


@hydra.main(version_base=None, config_path="../configs", config_name="ft_config")
def test_peft(cfg: DictConfig) -> None:
    AVAIL_GPUS = 0
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        AVAIL_GPUS = torch.cuda.device_count()
        print(f'There are {AVAIL_GPUS} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu") 

    test_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/PEFT_{cfg.ft_models.name}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}.csv'
    aggr_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/PEFT_{cfg.ft_models.name}-' \
                        f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-aggregate.csv'

    test_df = get_predictions(lora_module=cfg.experiments.lora_path,
                         test_path=cfg.datasets.test, 
                         batch_size=cfg.experiments.batch_size,
                         device=device, 
                         prefix=cfg.datasets.prefix,
                         src_lan=cfg.datasets.src_lan,
                         trg_lan=cfg.datasets.trg_lan,
                         limit=cfg.experiments.limit
                        )
    
    evaluator = Evaluator()
    evaluator.evaluating_from_dataframe(dataframe=test_df, save_path=test_save_path)
    aggregate_metrics_df = evaluator.calculating_corpus_metrics_from_dataframe(dataframe=test_df)
    aggregate_metrics_df.to_csv(aggr_save_path)

if __name__ == "__main__":
    test_peft()