from mt2magic.TestPEFTDataset import TestPEFTDataset
from mt2magic.FloresDataModule import FloresDataModule
from mt2magic.TranslatedDataModule import TranslatedDataModule
from mt2magic.PEFT_fine_tuner import PEFTModel
from mt2magic.evaluator import Evaluator

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
import torch
from transformers import pipeline

from omegaconf import DictConfig
import hydra
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
  This method generate the translations for the samples in the test set 
  using a fine-tuned model (fine-tuned with LoRA)
  and returns a pandas DataFrame that will be used to compute metrics
  Args:
    model (:obj) : model to be tested
    tokenizer (:obj) : tokenizer used
    test_path (str) : path to the csv file that contains the source sentences and the target sentences
    prefix (str) : prefix to append to the source sentence
    src_lan (str) : language of the source text
    trg_lan (str) : language of the target text
    batch_size (int) : dimension of the batches
    device (str) : device used for the inference (gpu or cpu)
    limit (int) : number of samples of the test set used for evaluation
  Returns: 
    df (pd.DataFrame) : pandas DataFrame with the source sentences, the target sentences, and the translations
"""
def get_predictions(model, 
                    tokenizer, 
                    test_path:str, 
                    prefix:str, 
                    src_lan:str, 
                    trg_lan:str, 
                    batch_size:int, 
                    device:str,
                    limit:int
                    ):
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
def fine_tuning(cfg: DictConfig) -> None:
    AVAIL_GPUS = 0
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        AVAIL_GPUS = torch.cuda.device_count()
        print(f'There are {AVAIL_GPUS} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
        accelerator = "gpu"
        use_quantization = True
                                                                                                                                                                                                                                                
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")   
        accelerator = "cpu"
        use_quantization = False
    
    wandb_logger = WandbLogger(name=f"{cfg.ft_models.name}_peft_{cfg.datasets.dataset}_{cfg.datasets.src_lan}_{cfg.datasets.trg_lan}", 
                               project='translated-challenge', 
                               entity='mt2magic', 
                               log_model=True
                               )
    if cfg.datasets.dataset == "translated":
        dm = TranslatedDataModule(train_file=cfg.datasets.train, 
                                val_file=cfg.datasets.dev, 
                                test_file=cfg.datasets.test,
                                tokenizer=cfg.ft_models.full_name, 
                                batch_size=cfg.experiments.batch_size,
                                max_length=cfg.experiments.max_length, 
                                prefix=cfg.datasets.prefix
                                )
    elif cfg.datasets.dataset == "flores":
        dm = FloresDataModule(dev_path=cfg.datasets.dev,
                        devtest_path=cfg.datasets.test,
                        tokenizer=cfg.ft_models.full_name, 
                        batch_size=cfg.experiments.batch_size,
                        max_length=cfg.experiments.max_length, 
                        prefix=cfg.datasets.prefix
                        )
    else:
        print("The selected dataset is not valid! Use Translated or Flores datasets")
        return
    
    dm.setup()
    model = PEFTModel(model_name=cfg.ft_models.full_name, 
                      lora_r=cfg.experiments.lora_r, 
                      lora_alpha=cfg.experiments.lora_alpha, 
                      lora_dropout=cfg.experiments.lora_dropout, 
                      device=device, 
                      lr=cfg.experiments.lr, 
                      use_quantization=use_quantization,
                      peft_mode=cfg.experiments.peft_mode
                    )
    
    # if more than one device add devices = AVAIL_GPUS and accumulate_grad_batches
    # for reproducibility add deterministic = True
    trainer = Trainer(
        max_epochs=cfg.experiments.num_epochs,
        accelerator = accelerator,
        devices = AVAIL_GPUS if AVAIL_GPUS else 1, # if we are not using GPUs set this to 1 anyway
        plugins=DeepSpeedPlugin(stage=3),
        accumulate_grad_batches=cfg.experiments.accumulate_grad_num,
        logger= wandb_logger
        )
    
    print("Fine-tuning...")

    trainer.fit(model, datamodule=dm)

    model.model.save_pretrained(f"models/{cfg.ft_models.name}_peft_{cfg.datasets.dataset}_{cfg.datasets.src_lan}_{cfg.datasets.trg_lan}/")

    print("Done with the fine-tuning!")
    
    print("Evaluation of the model...")

    test_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/PEFT_{cfg.ft_models.name}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}.csv'
    aggr_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/PEFT_{cfg.ft_models.name}-' \
                        f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-aggregate.csv'

    test_df = get_predictions(model.model, 
                              tokenizer=dm.tokenizer, 
                              test_path=cfg.datasets.test, 
                              prefix=cfg.datasets.prefix, 
                              batch_size=cfg.experiments.batch_size,
                              device=device,
                              src_lan=cfg.datasets.src_lan,
                              trg_lan=cfg.datasets.trg_lan,
                              limit=cfg.experiments.limit
                              )
    evaluator = Evaluator()
    evaluator.evaluating_from_dataframe(dataframe=test_df, save_path=test_save_path)
    aggregate_metrics_df = evaluator.calculating_corpus_metrics_from_dataframe(dataframe=test_df)
    aggregate_metrics_df.to_csv(aggr_save_path)

    print("Done with the evaluation!")

if __name__ == "__main__":
    fine_tuning()