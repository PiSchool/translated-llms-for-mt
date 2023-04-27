from mt2magic.evaluate.evaluator import Evaluator
from mt2magic.peft.TestPEFTDataset import TestPEFTDataset

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, T5Tokenizer, BloomTokenizerFast, TextGenerationPipeline, Text2TextGenerationPipeline
import pandas as pd

from omegaconf import DictConfig
import hydra

"""
  This method generate the translations for the samples in the test set 
  using a fine-tuned model (fine-tuned with LoRA)
  and returns a pandas DataFrame that will be used to compute metrics
  Args:
    lora_module (str) : path to the folder that contains the config file for LoRA
    model_type (str) : name of the model we are testing (bloom or t5)
    test_path (str) : path to the csv file that contains the source sentences and the target sentences
    device (str) : device used for the inference (gpu or cpu)
    src_lan (str) : language of the source text
    trg_lan (str) : language of the target text
    generate_config (:obj) : config file for the experiments
    limit (int) : number of samples of the test set used for evaluation
  Returns: 
    df (pd.DataFrame) : pandas DataFrame with the source sentences, the target sentences, and the translations
"""
def get_predictions(lora_module:str,
                    model_type:str,
                    test_path:str,
                    device:str, 
                    src_lan:str,
                    trg_lan:str,
                    generate_config,
                    limit:int=0,
                    ):
    config = PeftConfig.from_pretrained(lora_module)
    if "bloom" in model_type:
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = BloomTokenizerFast.from_pretrained(config.base_model_name_or_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, lora_module).to(device)
    model.eval()

    if "bloom" in model_type:
        translator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
    elif "t5" in model_type:
        translator = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
    else:
        raise Exception("The accepted model are: BLOOM and T5!")
    
    dataset = TestPEFTDataset(test_path, prompt_type=generate_config.prompt_type, src_lan=src_lan, trg_lan=trg_lan)
    results = []
    df = pd.read_csv(test_path)
    if limit > 0:
        df = df.iloc[:limit]
    elif limit == 0:
        limit = len(df)
    for translated_text in translator(dataset, 
                                    batch_size=generate_config.batch_size, 
                                    temperature=generate_config.temperature, 
                                    repetition_penalty=generate_config.repetition_penalty, 
                                    length_penalty=generate_config.length_penalty,
                                    do_sample=generate_config.do_sample,
                                    num_return_sequences=generate_config.num_return_sequences
                                    ):
        results.append(translated_text[0]["generated_text"])
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

    lora_path = f"models/{cfg.ft_models.name}_peft_{cfg.datasets.dataset}_{cfg.datasets.src_lan}_{cfg.datasets.trg_lan}/"
    
    test_df = get_predictions(lora_module=lora_path,
                            model_type=cfg.ft_models.name,
                            test_path=cfg.datasets.test, 
                            src_lan=cfg.datasets.languageA,
                            trg_lan=cfg.datasets.languageB,
                            device=device, 
                            limit=cfg.experiments.limit_test,
                            generate_config= cfg.experiments
                            )
    
    evaluator = Evaluator()
    evaluator.evaluating_from_dataframe(dataframe=test_df, save_path=test_save_path)
    aggregate_metrics_df = evaluator.calculating_corpus_metrics_from_dataframe(dataframe=test_df)
    aggregate_metrics_df.to_csv(aggr_save_path)

if __name__ == "__main__":
    test_peft()