import pandas as pd
import requests
from time import sleep

"""
This class is used for computing COMET score with Translated API.
"""
class Translated_Evaluator:

    def evaluating_from_dataframe(self, dataframe, src_lang, tgt_lang, gpus=1) -> dict:

        with open(f'source.{src_lang}', 'w', encoding="utf-8") as source_file:
            for sent in dataframe['source'].values:
                source_file.write(f'{sent}')

        with open('ref.txt', 'w', encoding="utf-8") as target_file:
            for sent in dataframe['target'].values:
                target_file.write(f'{sent}')

        with open(f'translation.{tgt_lang}', 'w', encoding="utf-8") as translation_file:
            for sent in dataframe['translation'].values:
                translation_file.write(f'{sent}')

        url_evaluate = 'http://mmt-evaluate.translatedlabs.com/evaluate'
        params_evaluate = {'gpus': gpus
            , 'source': f'./source.{src_lang}'
            , 'ref': './ref.txt'
            , 'hyp_0': f'translation.{tgt_lang}'}
        evaluate = requests.post(url_evaluate, data=params_evaluate).json()

        url_evaluation_status = 'http://mmt-evaluate.translatedlabs.com/evaluation_status'
        params_evaluation_status = {'task_id': evaluate['task_id']}
        evaluation_status = requests.post(url_evaluation_status, data=params_evaluation_status).json()

        eval_flag = True
        while eval_flag:
            sleep(3)
            if evaluation_status['status'] == 'Finished':
                eval_flag = False
                url_result = 'http://mmt-evaluate.translatedlabs.com/result'
                params_result = {'task_id': evaluate['task_id']}
                result = requests.post(url_result, data=params_result).json()

                bleu = result['bleu']['paired-ar']['systems'][0]['score']
                chrf = result['chrf']['paired-ar']['systems'][0]['score']
                comet_da = result['comet-da']['scoreA']

        return {'bleu': bleu, 'chrf': chrf, 'comet_da': comet_da}

    def evaluating_from_file(self, file_path, src_lang, tgt_lang
                             , gpus=1, separator=',', encoding='utf-8') -> dict:

        dataframe = pd.read_csv(file_path, sep=separator, encoding=encoding)

        with open(f'source.{src_lang}', 'w', encoding="utf-8") as source_file:
            for sent in dataframe['source'].values:
                source_file.write(f'{sent}')

        with open('ref.txt', 'w', encoding="utf-8") as target_file:
            for sent in dataframe['target'].values:
                target_file.write(f'{sent}')

        with open(f'translation.{tgt_lang}', 'w', encoding="utf-8") as translation_file:
            for sent in dataframe['translation'].values:
                translation_file.write(f'{sent}')

        url_evaluate = 'http://mmt-evaluate.translatedlabs.com/evaluate'
        params_evaluate = {'gpus': gpus
            , 'source': f'./source.{src_lang}'
            , 'ref': './ref.txt'
            , 'hyp_0': f'translation.{tgt_lang}'}
        evaluate = requests.post(url_evaluate, data=params_evaluate).json()

        url_evaluation_status = 'http://mmt-evaluate.translatedlabs.com/evaluation_status'
        params_evaluation_status = {'task_id': evaluate['task_id']}
        evaluation_status = requests.post(url_evaluation_status, data=params_evaluation_status).json()

        eval_flag = True
        while eval_flag:
            sleep(3)
            if evaluation_status['status'] == 'Finished':
                eval_flag = False
                url_result = 'http://mmt-evaluate.translatedlabs.com/result'
                params_result = {'task_id': evaluate['task_id']}
                result = requests.post(url_result, data=params_result).json()

                bleu = result['bleu']['paired-ar']['systems'][0]['score']
                chrf = result['chrf']['paired-ar']['systems'][0]['score']
                comet_da = result['comet-da']['scoreA']

        return {'bleu': bleu, 'chrf': chrf, 'comet_da': comet_da}
