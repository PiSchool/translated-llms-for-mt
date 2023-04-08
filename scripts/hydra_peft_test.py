from mt2magic.evaluator import Evaluator

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

from omegaconf import DictConfig
import hydra

def get_predictions(lora_module:str,
                    model_name:str, 
                    test_path:str, 
                    device:str, 
                    prefix:str
                    ):
    #config = PeftConfig.from_pretrained(lora_module)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,  load_in_8bit=True,  device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, lora_module, device_map='auto')
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

    df = get_predictions(lora_module=cfg.experiments.lora_path,
                         model_name=cfg.ft_models.full_name, 
                         test_path=cfg.datasets.test, 
                         device=device, 
                         prefix=cfg.datasets.prefix
                        )
    eval = Evaluator()
    df_translation = eval.evaluating_from_dataframe(df)

    corpus_bleu = eval.calculate_corpus_bleu(df_translation)
    mean_bleu = eval.calculate_mean_bleu(df_translation)
    corpus_chrf = eval.calculate_corpus_chrf(df_translation)
    mean_chrf = eval.calculate_mean_chrf(df_translation)
    mean_comet = eval.calculate_system_score_COMET(df_translation)
    print('*** *** ***')
    print(f'Corpus BLEU: {corpus_bleu}')
    print(f'Mean BLEU: {mean_bleu}')
    print('*** *** ***')
    print(f'Corpus chrf: {corpus_chrf}')
    print(f'Mean chrf: {mean_chrf}')
    print('*** *** ***')
    print(f'\nMean COMET: {mean_comet}')
    print('*** *** ***')
    return

if __name__ == "__main__":
    test_peft()