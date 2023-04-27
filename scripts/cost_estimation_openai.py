from mt2magic.utils.prompter import Prompter
from mt2magic.utils.openai_tokenizer import openaiTokenizer
from mt2magic.utils.make_cfg import prompter_cfg
import pandas as pd
from omegaconf import DictConfig
import hydra

"""
This script is used to estimate the cost of inference/finetuning for different openai models,
on the dataset that can be specified in the sweeper field of the Hydra config.
"""
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def estimate_cost(cfg: DictConfig) -> None:
    tokenizer = openaiTokenizer()
    test_df = pd.read_csv(cfg.datasets.test, sep=cfg.datasets.sep, encoding=cfg.datasets.encoding)
    train_df = pd.read_csv(cfg.datasets.dev, sep=cfg.datasets.sep, encoding=cfg.datasets.encoding)
    prompter = Prompter(prompter_cfg(cfg))
    prompts = []
    source_sentences = test_df["source"].tolist()
    for sent in source_sentences:
        prompts.append(prompter.get_prompt(sent))
    for model in tokenizer.inference_cost_per_token.keys():
        print(f"Model used: {model}")
        print(f"Dataset: {cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}")
        translation_cost = tokenizer.translation_cost(source_sentences, prompts, model)
        print(f"Translation cost: {translation_cost}")
    print(f"Model used: davinci fine-tuned")
    print(f"Dataset: {cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}")
    translation_cost = tokenizer.finetuned_translation_cost(source_sentences, prompts)
    print(f"Translation cost: {translation_cost}")
    train_sentences = train_df["source"].tolist()
    for sent in train_sentences:
        prompts.append(prompter.get_prompt(sent))
    finetune_cost = tokenizer.finetune_cost(train_sentences, prompts)
    print(f"Finetuning cost: {finetune_cost}")

if __name__ == "__main__":
    estimate_cost()