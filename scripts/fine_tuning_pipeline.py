from mt2magic.FloresDataModule import FloresDataModule
from mt2magic.TranslatedDataModule import TranslatedDataModule
from mt2magic.PEFT_fine_tuner import PEFTModel

import argparse
from pytorch_lightning import seed_everything, Trainer
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, T5Tokenizer
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--dataset')
args = parser.parse_args()

src_lang ="ita"
trg_lang = "spa"
model_path = "models/"
prefix = "Translate from Italian to Spanish:"

max_length = 64
lr = 3e-4
num_epochs = 1
batch_size = 1
accumulate_grad_num = 4

lora_alpha = 32
lora_dropout = 0.1
lora_r = 16

AVAIL_GPUS = 0
if torch.cuda.is_available():       
    device = torch.device("cuda")
    AVAIL_GPUS = torch.cuda.device_count()
    print(f'There are {AVAIL_GPUS} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
    accelerator = "gpu"
    quantization = True
                                                                                                                                                                                                                                            
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")   
    accelerator = "cpu"
    quantization = False

#f008a102b1eb1e581a8595aa2a0b66d20526ab1e
if args.model == "t5":
  model_name = "google/flan-t5-small"
  #model_name = "google/flan-t5-large"
  #model_name = "google/flan-ul2"
  tokenizer = T5Tokenizer.from_pretrained(model_name)
elif args.model == "bloom":
  model_name = "bigscience/mt0-small"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
else:
   print("Input for model not accepted, possible inputs are: 't5' or 'bloom'")

seed_everything(42)
if args.dataset == "flores":
    wandb_logger = WandbLogger(name=f"{args.model}_peft_Flores_{src_lang}_{trg_lang}", 
                               project='translated-challenge', 
                               entity='mt2magic', 
                               log_model=True, 
                               checkpoint_name="ciao"
                               )
    data_path = "data/external/flores200_dataset/"
    dm = FloresDataModule(src_lang,  
                        trg_lang,
                        data_path,
                        tokenizer=model_name, 
                        batch_size=batch_size,
                        max_length=max_length, 
                        prefix=prefix
                        )
elif args.dataset == "translated":
    wandb_logger = WandbLogger(name=f"{args.model}_peft_Translated_{src_lang}_{trg_lang}", project='translated-challenge', entity='mt2magic', log_model=True)
    train_path = "data/processed/translated-it-es-train.csv"
    val_path = "data/processed/translated-it-es-dev.csv"
    test_path = "data/processed/translated-it-es-test.csv"
    dm = TranslatedDataModule(train_path, 
                            val_path, 
                            test_path,
                            tokenizer=model_name, 
                            batch_size=batch_size,
                            max_length=max_length, 
                            prefix=prefix
                            )
else:
   print("Input for dataset not accepted, possible inputs are: 'flores' or 'translated'")

dm.setup()

wandb_logger.experiment.config["max_length"] = max_length
wandb_logger.experiment.config["lr"] = lr
wandb_logger.experiment.config["num_epochs"] = num_epochs
wandb_logger.experiment.config["batch_size"] = batch_size
wandb_logger.experiment.config["lora_alpha"] = lora_alpha
wandb_logger.experiment.config["lora_dropout"] = lora_dropout
wandb_logger.experiment.config["lora_r"] = lora_r
wandb_logger.experiment.config["accumulate_grad_num"] = accumulate_grad_num

model = PEFTModel(model_name, lora_r, lora_alpha, lora_dropout, device=device, lr=lr, quantization=quantization)

# if more than one device add devices = AVAIL_GPUS and accumulate_grad_batches
# for reproducibility add deterministic = True
trainer = Trainer(
    max_epochs=num_epochs,
    accelerator = accelerator,
    devices = AVAIL_GPUS if AVAIL_GPUS else 1, # if we are not using GPUs set this to 1 anyway
    accumulate_grad_batches=accumulate_grad_num,
    logger= wandb_logger,
    default_root_dir="models/",
    #callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=2)]
    )

trainer.fit(model, datamodule=dm)

trainer.save_checkpoint(f"models/{model_name}_peft_{args.dataset}_{src_lang}_{trg_lang}.ckpt")

print("Done with the fine-tuning!")