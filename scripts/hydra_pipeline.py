import pandas as pd
from mt2magic.gpt3_translator import gpt3_translator
from mt2magic.modernMT_translator import modernMT_translator
from mt2magic.evaluator import Evaluator
from mt2magic.make_cfg import prompter_cfg
from omegaconf import DictConfig
import hydra

"""`
Inference/evaluation pipeline that exploits Hydra configs. Set the combination of experiments
that you want to perform by modifying the sweeper in configs/config.yaml 
If performing a test of the pipeline, set the flag "test" to true, so that the experiments will 
process only the first 5 sentences of the dataset!
"""

test = True


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def translate_pipeline(cfg: DictConfig) -> None:
    test_df = pd.read_csv(cfg.datasets.test, sep=cfg.datasets.sep, encoding=cfg.datasets.encoding)
    if test:
        test_df = test_df.head()
    if cfg.experiments.model == "gpt":
        stop_seq = ['[target]', '[source]']
        gpt_param = {'temperature': 0.0, 'max_tokens': 256, 'stop': stop_seq}
        prompt_config = prompter_cfg(cfg)
        gpt_translator = gpt3_translator(API_KEY=cfg.experiments.key,
                                         prompt_config=prompt_config,
                                         model_name='davinci',
                                         param=gpt_param)
        gpt_translator.translate(test_df, prompter_config=prompt_config)
        test_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-' \
                         f'{cfg.experiments.strategy}-{cfg.experiments.n_shots}.csv'
        aggr_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-' \
                         f'{cfg.experiments.strategy}-{cfg.experiments.n_shots}-aggregate.csv'

    elif cfg.experiments.model == "gpt3_5":
        stop_seq = ['[target]', '[source]']
        gpt_param = {'temperature': 0.0, 'max_tokens': 256, 'stop': stop_seq}
        prompt_config = prompter_cfg(cfg)
        gpt_translator = gpt3_translator(API_KEY=cfg.experiments.key,
                                         prompt_config=prompt_config,
                                         model_name='text-davinci-003',
                                         param=gpt_param)
        gpt_translator.translate(test_df, prompter_config=prompt_config)
        test_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-' \
                         f'{cfg.experiments.strategy}-{cfg.experiments.n_shots}.csv'
        aggr_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-' \
                         f'{cfg.experiments.strategy}-{cfg.experiments.n_shots}-aggregate.csv'

    else:
        mt_translator = modernMT_translator(api_key=cfg.experiments.key,
                                            source_lang=cfg.datasets.src_lan,
                                            target_lang=cfg.datasets.trg_lan)
        mt_translator.translate(data=test_df)
        test_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}.csv'
        aggr_save_path = f'./data/processed/metrics/{cfg.datasets.dataset}/{cfg.experiments.model}-' \
                         f'{cfg.datasets.dataset}-{cfg.datasets.src_lan}-{cfg.datasets.trg_lan}-aggregate.csv'
    evaluator = Evaluator()
    evaluator.evaluating_from_dataframe(dataframe=test_df, save_path=test_save_path)
    aggregate_metrics_df = evaluator.calculating_corpus_metrics_from_dataframe(dataframe=test_df)
    aggregate_metrics_df.to_csv(aggr_save_path)


if __name__ == "__main__":
    translate_pipeline()
