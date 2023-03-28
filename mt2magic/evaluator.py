import logging

logging.disable(logging.CRITICAL)
import pandas as pd
import torch
import nltk

nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf, corpus_chrf
from comet import download_model, load_from_checkpoint

from nltk.tokenize import word_tokenize
import os


class Evaluator:

    def __init__(self, model_name='Unbabel/wmt22-comet-da'):

        self.COMET_model_path = download_model(model_name, saving_directory='./models/')

    def calculate_sentence_bleu(self, df_evaluation):
        """
            Calculating the sentence BLEU score for each translation.
        """
        df_evaluation['BLEU'] = 0
        smoothie = SmoothingFunction().method4
        for i, r in df_evaluation.iterrows():
            bleu_score = sentence_bleu([word_tokenize(str(r['target']))], word_tokenize(str(r['translation']))
                                       , smoothing_function=smoothie)
            df_evaluation.at[i, 'BLEU'] = bleu_score

        return df_evaluation

    def calculate_sentence_chrf(self, df_evaluation):
        """
            Calculating the sentence chrf score for each translation.
        """
        df_evaluation['chrf'] = 0
        for i, r in df_evaluation.iterrows():
            chrf_score = sentence_chrf((str(r['target'])), str(r['translation']))
            df_evaluation.at[i, 'chrf'] = chrf_score

        return df_evaluation

    def calculate_COMET(self, df_evaluation, batch_size=8, gpu_numbers=1):
        """
            Calculating the COMET score for each translation.
            model_name (:obj:`str`): Model name of COMET library from below link:
                1. https://huggingface.co/Unbabel
                The default value is 'Unbabel/wmt22-comet-da' which is built on top of XLM-R
                and has been trained on direct assessments from WMT17 to WMT20 and provides scores ranging from 0 to 1
                , where 1 represents a perfect translation.
                batch_size (:obj: 'int'): batch_size
                gpu_numbers (:obj: 'int'): Number of GPUs
        """
        if torch.cuda.is_available():
            gpu_numbers = gpu_numbers
        else:
            gpu_numbers = 0

        model = load_from_checkpoint(self.COMET_model_path)
        df_evaluation['COMET'] = 0
        for i, r in df_evaluation.iterrows():
            data = [
                {
                    'src': str(r['source']),
                    'mt': str(r['translation']),
                    'ref': str(r['target'])
                }
            ]
            model_output = model.predict(data, batch_size=batch_size, gpus=gpu_numbers)
            df_evaluation.at[i, 'COMET'] = model_output.scores[0]

        return df_evaluation

    def evaluating_from_dataframe(self, dataframe, save_path='/data/df_result_with_evaluation.csv'
                                  , COMET_model_batch_size=8, COMET_model_gpu_numbers=1):
        """
                    Evaluating translations from privided csv file path.
                    Keys and Values:
                        dataframe (:obj:`pandas dataframe'): Translation dataframe with agreed structure
                        save_path (:obj: 'str'): path for saving the result dataframe in csv format

                    Output:
                        dataframe (:obj: 'pandas dataframe'): The dataframe with 3 evaluation metrics columns (BLEU, chrf, COMET)
        """
        df_evaluation = dataframe.copy()
        df_evaluation = self.calculate_sentence_bleu(df_evaluation)
        df_evaluation = self.calculate_sentence_chrf(df_evaluation)
        df_evaluation = self.calculate_COMET(df_evaluation
                                             , batch_size=COMET_model_batch_size, gpu_numbers=COMET_model_gpu_numbers)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        df_evaluation.to_csv(save_path, sep=',')
        return df_evaluation

    def evaluating_from_file_path(self, prediction_file_path, sep=',', encoding='utf-8', save_path='/data/'
                                  , COMET_model_batch_size=8, COMET_model_gpu_numbers=1):
        """
                    Evaluating translations from privided csv file path.
                    Keys and Values:
                        prediction_file_path (:obj:`str'): CSV file path with agreed structure
                        sep (:obj: 'str'): seperator of csv file
                        encoding (:obj: 'str'): encoding of csv file
                        save_path (:obj: 'str'): path for saving the result dataframe in csv format

                    Output:
                        dataframe (:obj: 'pandas dataframe'): The dataframe with 3 evaluation metrics columns (BLEU, chrf, COMET)
        """

        df_evaluation = pd.read_csv(prediction_file_path, sep=sep, encoding=encoding)
        df_evaluation = self.calculate_sentence_bleu(df_evaluation)
        df_evaluation = self.calculate_sentence_chrf(df_evaluation)
        df_evaluation = self.calculate_COMET(df_evaluation
                                             , batch_size=COMET_model_batch_size, gpu_numbers=COMET_model_gpu_numbers)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        df_evaluation.to_csv(save_path, sep=',')
        return df_evaluation

    def calculate_corpus_bleu(self, df_evaluation):
        """
            Calculating the corpus BLEU score over entire translations.
        """
        list_of_references = []
        for sentence in df_evaluation['target'].values:
            list_of_references.append([word_tokenize(str(sentence))])

        hypotheses = []
        for sentence in df_evaluation['translation'].values:
            hypotheses.append(word_tokenize(str(sentence)))

        smoothie = SmoothingFunction().method4
        return corpus_bleu(list_of_references, hypotheses, smoothing_function=smoothie)

    def calculate_mean_bleu(self, df_evaluation):
        """
            Calculating the mean BLEU score over entire translations.
        """
        mean_bleu = df_evaluation.loc[:, 'BLEU'].mean()
        return mean_bleu

    def calculate_corpus_chrf(self, df_evaluation):
        """
            Calculating the corpus chrf score over entire translations.
        """
        list_of_references = []
        for sentence in df_evaluation['target'].values:
            list_of_references.append([str(sentence)])

        hypotheses = []
        for sentence in df_evaluation['translation'].values:
            hypotheses.append([str(sentence)])

        return corpus_chrf(list_of_references, hypotheses)

    def calculate_mean_chrf(self, df_evaluation):
        """
            Calculating the mean chrf score over entire translations.
        """
        mean_bleu = df_evaluation.loc[:, 'chrf'].mean()
        return mean_bleu

    def calculate_system_score_COMET(self, df_evaluation, batch_size=256, gpu_numbers=1):
        """
            Calculate system_score (mean) COMET score over entire translations.
            Keys and Values:
                df_prediction (:obj:`pandas dataframe'): Dataframe contains source text, reference text ,and translation text
                model_name (:obj:`str`): Model name of COMET library from below link:
                1. https://huggingface.co/Unbabel
                The default value is 'Unbabel/wmt22-comet-da' which is built on top of XLM-R
                and has been trained on direct assessments from WMT17 to WMT20 and provides scores ranging from 0 to 1
                , where 1 represents a perfect translation.
                batch_size (:obj: 'int'): batch_size
                gpu_numbers (:obj: 'int'): Number of GPUs

            Output:
                system_score (:obj: 'float'): The mean COMET score over entire translations.
        """
        if torch.cuda.is_available():
            gpu_numbers = gpu_numbers
        else:
            gpu_numbers = 0

        model = load_from_checkpoint(self.COMET_model_path)

        data_list = []
        for i, r in df_evaluation.iterrows():
            data = {
                'src': str(r['source']),
                'mt': str(r['translation']),
                'ref': str(r['target'])
            }
            data_list.append(data)

        model_output = model.predict(data_list, batch_size=batch_size, gpus=gpu_numbers)
        return model_output.system_score
