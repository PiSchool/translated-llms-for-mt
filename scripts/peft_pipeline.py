from mt2magic.TestPEFTDataset import TestPEFTDataset
#from mt2magic.FloresDataModule import FloresDataModule
from mt2magic.TranslatedDataModule import TranslatedDataModule
from mt2magic.PEFT_fine_tuner import PEFTModel
from mt2magic.evaluator import Evaluator

import pandas as pd
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
from transformers import TextGenerationPipeline, Text2TextGenerationPipeline

from omegaconf import DictConfig, OmegaConf
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
    model_type (str) : name of the model we are testing (bloom or t5)
    test_path (str) : path to the csv file that contains the source sentences and the target sentences
    device (str) : device used for the inference (gpu or cpu)
    generate_config (:obj) : config file for the experiments
    src_lan (str) : language of the source text
    trg_lan (str) : language of the target text
    limit (int) : number of samples of the test set used for evaluation
  Returns: 
    df (pd.DataFrame) : pandas DataFrame with the source sentences, the target sentences, and the translations
"""
def get_predictions(model,
                    tokenizer,
                    model_type:str,
                    test_path:str,
                    device:str, 
                    generate_config,
                    src_lan:str,
                    trg_lan:str,
                    limit:int=0,
                    ):
    model.eval()

    if "bloom" in model_type:
        translator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
    elif "t5" in model_type:
        translator = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
    else:
        raise Exception("The accepted model are: bloom and t5!")
    
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
def fine_tuning(cfg: DictConfig) -> None:
    AVAIL_GPUS = 0
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        AVAIL_GPUS = torch.cuda.device_count()
        print(f'There are {AVAIL_GPUS} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
        accelerator = "gpu"
        use_quantization = cfg.experiments.quantization                                                                                                                                                                                                                                       
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
                                num_workers=cfg.experiments.num_workers,
                                max_length=cfg.experiments.max_length, 
                                prefix_type=cfg.experiments.prompt_type,
                                src_lan=cfg.datasets.languageA,
                                trg_lan=cfg.datasets.languageB,
                                limit=cfg.experiments.limit_train
                                )
    # elif cfg.datasets.dataset == "flores":
    #     dm = FloresDataModule(dev_path=cfg.datasets.dev,
    #                     devtest_path=cfg.datasets.test,
    #                     tokenizer=cfg.ft_models.full_name, 
    #                     batch_size=cfg.experiments.batch_size,
    #                     max_length=cfg.experiments.max_length, 
    #                     prefix=cfg.datasets.prefix
    #                     )
    else:
        raise Exception("The selected dataset is not valid! Use Translated datasets")
    
    wandb_logger.experiment.config.update = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

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
    
    if accelerator=="cpu":
        deepspeed_strategy = "ddp"
        precision = "bf16-mixed"
    else:
        if not use_quantization:
            precision = cfg.experiments.precision
        else:
            precision = "32-true"

        if cfg.experiments.strategy == 2:
            deepspeed_strategy = "deepspeed_stage_3"
        else:
            deepspeed_strategy = "deepspeed_stage_3"

    trainer = L.Trainer(
        max_epochs=cfg.experiments.num_epochs,
        accelerator = accelerator,
        devices = AVAIL_GPUS if AVAIL_GPUS else 1, # if we are not using GPUs set this to 1 anyway
        strategy=deepspeed_strategy, 
        precision=precision,
        accumulate_grad_batches=cfg.experiments.accumulate_grad_num,
        logger= wandb_logger
        )
    
    print("Fine-tuning...")

    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    model.model.save_pretrained(f"models/{cfg.ft_models.name}_peft_{cfg.datasets.dataset}_{cfg.datasets.src_lan}_{cfg.datasets.trg_lan}/")

    print("Done with the fine-tuning!")
    
    print("Evaluation of the model...")

    test_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/PEFT_{cfg.ft_models.name}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}.csv'
    aggr_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/PEFT_{cfg.ft_models.name}-' \
                        f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-aggregate.csv'

    test_df = get_predictions(model.model, 
                            model_type=cfg.ft_models.name, 
                            tokenizer=dm.tokenizer, 
                            test_path=cfg.datasets.test,  
                            device=device,
                            limit=cfg.experiments.limit_test,
                            generate_config=cfg.experiments,
                            src_lan=cfg.datasets.languageA,
                            trg_lan=cfg.datasets.languageB
                            )
    evaluator = Evaluator()
    evaluator.evaluating_from_dataframe(dataframe=test_df, save_path=test_save_path)
    aggregate_metrics_df = evaluator.calculating_corpus_metrics_from_dataframe(dataframe=test_df)
    aggregate_metrics_df.to_csv(aggr_save_path)

    print("Done with the evaluation!")

if __name__ == "__main__":
    fine_tuning()