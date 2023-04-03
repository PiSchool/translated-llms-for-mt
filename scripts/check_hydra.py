import pandas as pd
from mt2magic.gpt3_translator import gpt3_translator
from mt2magic.modernMT_translator import modernMT_translator
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def translate_pipeline(cfg: DictConfig) -> None:
    test_df = pd.read_csv(cfg.datasets.test, sep=cfg.datasets.sep, encoding=cfg.datasets.encoding)
    print(cfg.datasets.sep)
    print(test_df["source"].head())
    if cfg.experiments.model == "gpt":
        stop_seq = ['[target]', '[source]']
        gpt_param = {'temperature': 0.0, 'max_tokens': 256,'stop': stop_seq}
        prompt_config = {'n_shots': cfg.experiments.n_shots,
                         'strategy': cfg.experiments.strategy,
                         'pool_path': cfg.datasets.dev,
                         'embeddings_path': cfg.datasets.emb_src,
                         'sep': cfg.datasets.sep,
                         'encoding': cfg.datasets.encoding}
        gpt_translator = gpt3_translator(API_KEY=cfg.experiments.key,
                                         prompt_config=prompt_config,
                                         model_name='davinci',
                                        param=gpt_param)
        #gpt_translator.translate(test_df, prompter_config=prompt_config)
        test_save_path = f'./data/processed/metrics/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-' \
                         f'{cfg.experiments.strategy}-{cfg.experiments.n_shots}.csv'
        aggr_save_path = f'./data/processed/metrics/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-' \
                         f'{cfg.experiments.strategy}-{cfg.experiments.n_shots}-aggregate.csv'

    else:
        mt_translator = modernMT_translator(api_key=cfg.experiments.key,
                                            source_lang=cfg.datasets.src_lan,
                                            target_lang=cfg.datasets.trg_lan)
        #mt_translator.translate(data=test_df)
        test_save_path = f'./data/processed/metrics/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}.csv'
        aggr_save_path = f'./data/processed/metrics/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-aggregate.csv'



if __name__ == "__main__":
    translate_pipeline()