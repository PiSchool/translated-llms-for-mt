from mt2magic.flores_dataset import FloresDataset

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

src_path = 'data/external/flores200_dataset/dev/ita_Latn.dev'
trg_path = 'data/external/flores200_dataset/dev/spa_Latn.dev'
src_eval_path = 'data/external/flores200_dataset/devtest/ita_Latn.devtest'
trg_eval_path = 'data/external/flores200_dataset/devtest/spa_Latn.devtest'
model_path = "models/"
prefix = "Translate from Italian to Spanish: "

m = "t5"

max_length = 256
lr = 1e-3
num_epochs = 1
batch_size = 1

AVAIL_GPUS = 0
if torch.cuda.is_available():       
    device = torch.device("cuda")
    AVAIL_GPUS = torch.cuda.device_count()
    print(f'There are {AVAIL_GPUS} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
                                                                                                                                                                                                                                            
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")   

if m == "t5":
  #model_name = "google/flan-t5-small"
  model_name = "google/mt5-small"
  #model_name = "google/flan-ul2"
  tokenizer = T5Tokenizer.from_pretrained(model_name)
elif m == "bloom":
  model_name = "bigscience/mt0-small"
  tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = FloresDataset(src_path, trg_path, tokenizer, prefix)
eval_dataset = FloresDataset(src_eval_path, trg_eval_path, tokenizer, prefix)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
peft_model = get_peft_model(model, peft_config)

optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
#config.optimizer = "AdamW"

# training and evaluation
peft_model = peft_model.to(device)
#wandb.watch(peft_model, log="all")
print("Starting training...")
for epoch in range(num_epochs):
    peft_model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = peft_model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    peft_model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = peft_model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
    #wandb.log({'epoch': epoch + 1, 'train_loss': train_epoch_loss, 'eval_loss':eval_epoch_loss})
print("Done!")

# saving model
model_id = f"{model_path}{peft_config.peft_type}_{m}"
print("Saving model in {}".format(model_id))
peft_model.save_pretrained(model_id)

