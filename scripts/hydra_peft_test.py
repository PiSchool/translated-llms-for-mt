from mt2magic.evaluator import Evaluator

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

from omegaconf import DictConfig
import hydra

def get_predictions(lora_module:str,
                    test_path:str, 
                    device:str, 
                    prefix:str
                    ):
    config_path = lora_module + "adapter_config.json"
    config = PeftConfig.from_pretrained(lora_module)
    # Loading the model in int8 is going to slow down inference for some reason
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)#,  load_in_8bit=True,  device_map='auto')
    model = PeftModel.from_pretrained(model, lora_module).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model.eval()

    data = pd.read_csv(test_path)
    results = []
    for _,s in tqdm(data.iterrows(), total=data.shape[0]):
        message = prefix + s["source"]
        inputs = tokenizer.encode(message, return_tensors="pt", padding=True).to(device)
        output = model.generate(inputs=inputs, max_length=512)
        results.append([s["source"], s["target"], tokenizer.decode(output[0], skip_special_tokens =True)])

    df = pd.DataFrame(results, columns=["source","target","translation"])
    return df

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
                         device=device, 
                         prefix=cfg.datasets.prefix
                        )
    evaluator = Evaluator()
    evaluator.evaluating_from_dataframe(dataframe=test_df, save_path=test_save_path)
    aggregate_metrics_df = evaluator.calculating_corpus_metrics_from_dataframe(dataframe=test_df)
    aggregate_metrics_df.to_csv(aggr_save_path)

if __name__ == "__main__":
    test_peft()