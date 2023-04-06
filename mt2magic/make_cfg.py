from mt2magic.formatting import PromptConfig, Parameters
from omegaconf import DictConfig
def prompter_cfg(cfg: DictConfig) -> PromptConfig:
    prompt_config = {'n_shots': cfg.experiments.n_shots,
                     'strategy': cfg.experiments.strategy,
                     'pool_path': cfg.datasets.dev,
                     'embeddings_path': cfg.datasets.emb_src,
                     'sep': cfg.datasets.sep,
                     'encoding': cfg.datasets.encoding}
    return prompt_config