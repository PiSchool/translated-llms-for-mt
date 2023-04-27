import torch
import lightning as L
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training

"""
Class to fine-tune models using LoRA
Args:
  model_name (str) : name used to load a model from Hugging Face
  lora_r (int) : rank of the matrixes in LoRA
  lora_alpha (float) : the alpha parameter for LoRA scaling
  lora_dropout (float) : a float that represents the dropout rate for the LoRA regularization
  device (str) : which kind of device is used to train, "cuda" or "cpu"
  lr (float) : learning rate
  use_quantization (bool) : True to use quantization to int8 (possible if device is gpu)
  peft_mode (bool) : True to use LoRA, False to fine-tune the model in a "classic" way
"""
class PEFTModel(L.LightningModule):
  def __init__(self, 
              model_name:str, 
              lora_r:int, 
              lora_alpha:float, 
              lora_dropout:float, 
              device:str, 
              lr:float=1e-3, 
              use_quantization:bool=False,
              peft_mode:bool=True
              ):
    super().__init__()
    
    if use_quantization:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto')
      model = prepare_model_for_int8_training(model)
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if peft_mode:
      self.peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                                  inference_mode=False, 
                                  target_modules=["q", "v"], 
                                  r=lora_r, 
                                  lora_alpha=lora_alpha, 
                                  lora_dropout=lora_dropout
                                  )
      self.model = get_peft_model(model, self.peft_config).to(device)
    else:
      self.model = model.to(device)
    
    self.lr = lr
    self.save_hyperparameters()

  def forward(self, **inputs):
    return self.model(**inputs)

  def predict_step(self, batch, batch_idx:int, dataloader_idx:int=0):
    return self(**batch)

  def training_step(self, batch, batch_idx:int):
    outputs = self(**batch)
    loss = outputs.loss
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx:int):
    outputs = self(**batch)
    loss = outputs.loss
    self.log('val_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    return optimizer