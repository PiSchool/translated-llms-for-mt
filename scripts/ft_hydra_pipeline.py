from mt2magic.FloresDataModule import FloresDataModule
from mt2magic.TranslatedDataModule import TranslatedDataModule
from mt2magic.PEFT_fine_tuner import PEFTModel

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, T5Tokenizer
import torch

from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="config")
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
    
    wandb_logger = WandbLogger(name=f"{cfg.ft_models}_peft_{cfg.datasets}_{cfg.datasets.src_lan}_{cfg.datasets.trg_lan}", 
                               project='translated-challenge', 
                               entity='mt2magic', 
                               log_model=True
                               )
    
    dm = TranslatedDataModule(cfg.datasets.train, 
                            cfg.datasets.dev, 
                            cfg.datasets.test,
                            tokenizer=cfg.ft_models, 
                            batch_size=cfg.experiments.batch_size,
                            max_length=cfg.experiments.max_length, 
                            prefix=cfg.datasets.prefix
                            )
    dm.setup()

    model = PEFTModel(cfg.ft_models.full_name, 
                      cfg.experiments.lora_r, 
                      cfg.experiments.lora_alpha, 
                      cfg.experiments.lora_dropout, 
                      device=device, 
                      lr=cfg.experiments.lr, 
                      use_quantization=use_quantization
                    )
    
    # if more than one device add devices = AVAIL_GPUS and accumulate_grad_batches
    # for reproducibility add deterministic = True
    trainer = Trainer(
        max_epochs=cfg.experiments.num_epochs,
        accelerator = accelerator,
        devices = AVAIL_GPUS if AVAIL_GPUS else 1, # if we are not using GPUs set this to 1 anyway
        accumulate_grad_batches=cfg.experiments.accumulate_grad_num,
        logger= wandb_logger,
        default_root_dir="models/",
        #callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=2)]
        )
    
    trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(f"models/{cfg.ft_models}_peft_{cfg.datasets}_{cfg.datasets.src_lang}_{cfg.datasets.trg_lang}.ckpt")

    print("Done with the fine-tuning!")



if __name__ == "__main__":
    fine_tuning()