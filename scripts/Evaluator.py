import os
import logging
logging.disable(logging.CRITICAL)
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.chrf_score import sentence_chrf, corpus_chrf
from nltk.tokenize import word_tokenize
from comet import download_model, load_from_checkpoint


class Evaluator:

    def calculate_sentence_bleu(self, df_evaluation):
        """
            Calculating the sentence BLEU score for each translation.
        """
        df_evaluation['BLEU'] = 0
        for i, r in df_evaluation.iterrows():
            bleu_score = sentence_bleu([word_tokenize(r['target'])], word_tokenize(r['translation']))
            df_evaluation.at[i, 'BLEU'] = bleu_score

        return df_evaluation

    def calculate_sentence_chrf(self, df_evaluation):
        """
            Calculating the sentence chrf score for each translation.
        """
        df_evaluation['chrf'] = 0
        for i, r in df_evaluation.iterrows():
            chrf_score = sentence_chrf((r['target']), r['translation'])
            df_evaluation.at[i, 'chrf'] = chrf_score

        return df_evaluation

    def calculate_COMET(self, df_evaluation, model_name='Unbabel/wmt22-comet-da'
                                     , batch_size=8, gpu_numbers=1):
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
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        df_evaluation['COMET'] = 0
        for i, r in df_evaluation.iterrows():
            data = [
                {
                    'src': r['source'],
                    'mt': r['translation'],
                    'ref': r['target']
                }
            ]
            model_output = model.predict(data, batch_size=batch_size, gpus = gpu_numbers)
            df_evaluation.at[i, 'COMET'] = model_output.scores[0]

        return df_evaluation

    def evaluating_from_dataframe(self, dataframe, save_path='/data/'):
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
        df_evaluation = self.calculate_COMET(df_evaluation)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path += 'df_prediction_with_BLEU'
        df_evaluation.to_csv(save_path, sep=',')
        return df_evaluation

    def evaluating_from_file_path(self, prediction_file_path, sep=',', encoding='utf-8', save_path='/data/'):
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
        df_evaluation = self.calculate_COMET(df_evaluation)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path += 'df_prediction_with_BLEU'
        df_evaluation.to_csv(save_path, sep=',')
        return df_evaluation

    def calculate_corpus_bleu(self, df_evaluation):
        """
            Calculating the corpus BLEU score over entire translations.
        """
        list_of_references = []
        for sentence in df_evaluation['target'].values:
            list_of_references.append([word_tokenize(sentence)])

        hypotheses = []
        for sentence in df_evaluation['translation'].values:
            hypotheses.append(word_tokenize(sentence))

        return corpus_bleu(list_of_references, hypotheses)

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
            list_of_references.append([sentence])

        hypotheses = []
        for sentence in df_evaluation['translation'].values:
            hypotheses.append([sentence])

        return corpus_chrf(list_of_references, hypotheses)

    def calculate_mean_chrf(self, df_evaluation):
        """
            Calculating the mean chrf score over entire translations.
        """
        mean_bleu = df_evaluation.loc[:, 'chrf'].mean()
        return mean_bleu

    def calculate_system_score_COMET(self, df_evaluation, model_name='Unbabel/wmt22-comet-da'
                                     , batch_size=256, gpu_numbers=1):
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

        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)

        data_list = []
        for i, r in df_evaluation.iterrows():
            data = {
                'src': r['source'],
                'mt': r['translation'],
                'ref': r['target']
            }
            data_list.append(data)

        model_output = model.predict(data_list, batch_size=batch_size, gpus=gpu_numbers)
        return model_output.system_score
