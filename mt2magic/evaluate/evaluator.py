import logging
import pandas as pd
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf, corpus_chrf
from comet import download_model, load_from_checkpoint
from nltk.tokenize import word_tokenize

nltk.download('punkt')
logging.disable(logging.CRITICAL)

"""
This class is used for evaluating translations. We've implemented BLEU, ChrF, and COMET scores in this class.
"""
class Evaluator:

    def __init__(self, model_name='Unbabel/wmt22-comet-da', run_without_gpu=False):

        self.COMET_sytem_score = None
        self.run_without_gpu = run_without_gpu

        if torch.cuda.is_available():
            self.COMET_model_path = download_model(model_name, saving_directory='./models/')
        else:
            if self.run_without_gpu:
                self.COMET_model_path = download_model(model_name, saving_directory='./models/')
            else:
                print('Skipping COMET score model downloading due to the absence of GPU ...'
                      '\nIf you want to download it with CPU then set run_without_gpu=True')

    def calculate_sentence_bleu(self, dataframe: pd.DataFrame):
        """
        Calculating the sentence BLEU score for each translation.
        """
        dataframe['BLEU2'] = 0
        dataframe['BLEU3'] = 0
        dataframe['BLEU4'] = 0
        smoothie = SmoothingFunction().method4
        weights = [
            (1. / 2., 1. / 2.),
            (1. / 3., 1. / 3., 1. / 3.),
            (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
        ]
        for i, r in dataframe.iterrows():
            bleu_scores = sentence_bleu([word_tokenize(str(r['target']))], word_tokenize(str(r['translation']))
                                        , weights, smoothing_function=smoothie)

            dataframe.at[i, 'BLEU2'] = bleu_scores[0]
            dataframe.at[i, 'BLEU3'] = bleu_scores[1]
            dataframe.at[i, 'BLEU4'] = bleu_scores[2]

        return dataframe

    def calculate_sentence_chrf(self, dataframe: pd.DataFrame):
        """
        Calculating the sentence chrf score for each translation.
        """
        dataframe['chrf'] = 0
        for i, r in dataframe.iterrows():
            chrf_score = sentence_chrf((str(r['target'])), str(r['translation']))
            dataframe.at[i, 'chrf'] = chrf_score

        return dataframe

    def calculate_COMET(self, dataframe: pd.DataFrame, batch_size=16, gpu_numbers=1):
        """
        Calculating the COMET score for each translation and also COMET sytem_score for entire translations.
        Args
            batch_size (:obj: 'int'): batch_size
            gpu_numbers (:obj: 'int'): Number of GPUs
        Returns
            dataframe with added COMET score
        """
        if torch.cuda.is_available():
            gpu_numbers = gpu_numbers
        else:
            gpu_numbers = 0

        model = load_from_checkpoint(self.COMET_model_path)
        data_list = []
        for i, r in dataframe.iterrows():
            data = {
                'src': str(r['source']),
                'mt': str(r['translation']),
                'ref': str(r['target'])
            }
            data_list.append(data)

        model_output = model.predict(data_list, batch_size, gpu_numbers)
        dataframe['COMET'] = model_output.scores

        # You can comment this line if you don't have memory problems
        del model

        # Add COMET system_score to self.COMET_sytem_score variable 
        # so when we need COMET system_score, there won't be any need to recalculate it
        self.COMET_sytem_score = model_output.system_score

        return dataframe

    def evaluating_from_dataframe(self, dataframe: pd.DataFrame, save_path='/data/df_result_with_evaluation.csv'
                                  , COMET_model_batch_size=8, COMET_model_gpu_numbers=1):
        """
        Evaluating translations from privided csv file path.
        Args
            dataframe (:obj:`pandas dataframe'): Translation dataframe with agreed structure
            save_path (:obj: 'str'): path for saving the result dataframe in csv format

        Returns
            dataframe (:obj: 'pandas dataframe'): The dataframe with 3 evaluation metrics columns (BLEU, chrf, COMET)
        """
        dataframe = self.calculate_sentence_bleu(dataframe)
        dataframe = self.calculate_sentence_chrf(dataframe)
        if torch.cuda.is_available():
            dataframe = self.calculate_COMET(dataframe, batch_size=COMET_model_batch_size,
                                             gpu_numbers=COMET_model_gpu_numbers)
        else:
            if self.run_without_gpu:
                dataframe = self.calculate_COMET(dataframe, batch_size=COMET_model_batch_size,
                                                 gpu_numbers=COMET_model_gpu_numbers)
            else:
                print('Skipping COMET score calculation due to the absence of GPU ...'
                      '\nIf you want to calculate it with CPU then set run_without_gpu=True')

        dataframe.to_csv(save_path, sep=',')
        return dataframe

    def evaluating_from_file_path(self, prediction_file_path: str, sep=',', encoding='utf-8', save_path='/data/'
                                  , COMET_model_batch_size=8, COMET_model_gpu_numbers=1):
        """
        Evaluating translations from privided csv file path.
        Args
            prediction_file_path (:obj:`str'): CSV file path with agreed structure
            sep (:obj: 'str'): seperator of csv file
            encoding (:obj: 'str'): encoding of csv file
            save_path (:obj: 'str'): path for saving the result dataframe in csv format

        Returns
            dataframe (:obj: 'pandas dataframe'): The dataframe with 3 evaluation metrics columns (BLEU, chrf, COMET)
        """

        dataframe = pd.read_csv(prediction_file_path, sep=sep, encoding=encoding)
        dataframe = self.calculate_sentence_bleu(dataframe)
        dataframe = self.calculate_sentence_chrf(dataframe)
        if torch.cuda.is_available():
            dataframe = self.calculate_COMET(dataframe, batch_size=COMET_model_batch_size,
                                             gpu_numbers=COMET_model_gpu_numbers)
        else:
            if self.run_without_gpu:
                dataframe = self.calculate_COMET(dataframe, batch_size=COMET_model_batch_size,
                                                 gpu_numbers=COMET_model_gpu_numbers)
            else:
                print('Skipping COMET score calculation due to the absence of GPU ...'
                      '\nIf you want to calculate it with CPU then set run_without_gpu=True')

        dataframe.to_csv(save_path, sep=',')
        return dataframe

    def calculating_corpus_metrics_from_dataframe(self, dataframe: pd.DataFrame):
        corpus_BLEUs = self.calculate_corpus_bleu(dataframe)
        corpus_chrf = self.calculate_corpus_chrf(dataframe)
        try:
            corpus_COMET = self.get_system_score_COMET()
        except ValueError:
            if torch.cuda.is_available():
                corpus_COMET = self.calculate_system_score_COMET(dataframe)
            else:
                if self.run_without_gpu:
                    corpus_COMET = self.calculate_system_score_COMET(dataframe)
                else:
                    corpus_COMET = None
                    print('Skipping COMET score calculation due to the absence of GPU ...'
                          '\nIf you want to calculate it with CPU then set run_without_gpu=True')

        data = {'BLEU2': corpus_BLEUs[0], 'BLEU3': corpus_BLEUs[1], 'BLEU4': corpus_BLEUs[2]
            , 'corpus_chrf': corpus_chrf, 'corpus_COMET': corpus_COMET}
        df_result = pd.DataFrame([data])

        return df_result

    def calculate_corpus_bleu(self, dataframe: pd.DataFrame):
        """
        Calculating the corpus BLEU score over entire translations.
        Args
            dataframe (:obj:`pandas dataframe`):
        Return
            dictionary (:obj: `dict`): dictionary of BLEU2, BLEU3, and BLEU4 scores
        """
        list_of_references = []
        for sentence in dataframe['target'].values:
            list_of_references.append([word_tokenize(str(sentence))])

        hypotheses = []
        for sentence in dataframe['translation'].values:
            hypotheses.append(word_tokenize(str(sentence)))

        weights = [
            (1. / 2., 1. / 2.),
            (1. / 3., 1. / 3., 1. / 3.),
            (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
        ]
        smoothie = SmoothingFunction().method4
        bleu_corpus_scores = corpus_bleu(list_of_references, hypotheses, weights, smoothing_function=smoothie)
        return bleu_corpus_scores

    def calculate_mean_bleu(self, dataframe: pd.DataFrame):
        """
            Calculating the mean BLEU score over entire translations.
        """
        mean_bleu = dataframe.loc[:, 'BLEU'].mean()
        return mean_bleu

    def calculate_corpus_chrf(self, dataframe: pd.DataFrame):
        """
        Calculating the corpus chrf score over entire translations.
        """
        list_of_references = []
        for sentence in dataframe['target'].values:
            list_of_references.append([str(sentence)])

        hypotheses = []
        for sentence in dataframe['translation'].values:
            hypotheses.append([str(sentence)])

        return corpus_chrf(list_of_references, hypotheses)

    def calculate_mean_chrf(self, dataframe: pd.DataFrame):
        """
        Calculating the mean chrf score over entire translations.
        """
        mean_bleu = dataframe.loc[:, 'chrf'].mean()
        return mean_bleu

    def get_system_score_COMET(self):
        if self.COMET_sytem_score is None:
            raise ValueError('COMET system score has not been computed yet!\nEvaluate dataframe first using '
                             'evaluating_from_file_path() or '
                             'evaluating_from_dataframe() or\ncall calculate_system_score_COMET() to '
                             'compute it directly.')
        else:
            return self.COMET_sytem_score

    def calculate_system_score_COMET(self, dataframe: pd.DataFrame, batch_size=16, gpu_numbers=1):
        """
        Calculate system_score (mean) COMET score over entire translations.
        Args
            df_prediction (:obj:`pandas dataframe'): Dataframe contains source text, reference text ,and translation text
            model_name (:obj:`str`): Model name of COMET library from below link:
            1. https://huggingface.co/Unbabel
            The default value is 'Unbabel/wmt22-comet-da' which is built on top of XLM-R
            and has been trained on direct assessments from WMT17 to WMT20 and provides scores ranging from 0 to 1
            , where 1 represents a perfect translation.
            batch_size (:obj: 'int'): batch_size
            gpu_numbers (:obj: 'int'): Number of GPUs

        Returns
            system_score (:obj: 'float'): The mean COMET score over entire translations.
        """
        if torch.cuda.is_available():
            gpu_numbers = gpu_numbers
        else:
            gpu_numbers = 0

        model = load_from_checkpoint(self.COMET_model_path)

        data_list = []
        for i, r in dataframe.iterrows():
            data = {
                'src': str(r['source']),
                'mt': str(r['translation']),
                'ref': str(r['target'])
            }
            data_list.append(data)

        model_output = model.predict(data_list, batch_size=batch_size, gpus=gpu_numbers)
        system_comet = model_output.system_score

        # You can comment this line if you don't have memory problems
        del model, data_list, data, model_output

        return system_comet

    def set_run_without_gpu(self, run_without_gpu):
        self.run_without_gpu = run_without_gpu
