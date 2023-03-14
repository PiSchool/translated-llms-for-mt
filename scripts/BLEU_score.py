import pandas as pd
import nltk

nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
import os


class Evaluator():

    def __init__(
            self,
            source_lang,
            target_lang
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def calculate_bleu_score(self, dataframe, save_path='/data/'):
        df_prediction = dataframe.copy()
        df_prediction['BLEU'] = 0
        for i, r in df_prediction.iterrows():
            bleu_score = sentence_bleu([word_tokenize(r['target'])], word_tokenize(r['translation']))
            df_prediction.at[i, 'BLEU'] = bleu_score

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path += 'df_prediction_with_BLEU'
        df_prediction.to_csv(save_path, sep=',')
        return df_prediction

    def calculate_bleu_score(self, prediction_file_path, sep=',', encoding='utf-8', save_path='/data/'):
        df_prediction = pd.read_csv(prediction_file_path, sep=sep, encoding=encoding)
        df_prediction['BLEU'] = 0
        for i, r in df_prediction.iterrows():
            bleu_score = sentence_bleu([word_tokenize(r['target'])], word_tokenize(r['translation']))
            df_prediction.at[i, 'BLEU'] = bleu_score

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path += 'df_prediction_with_BLEU'
        df_prediction.to_csv(save_path, sep=',')
        return df_prediction

    def calculate_mean_bleu(self, df_prediction):
        mean_bleu = df_prediction.loc[:, 'BLEU'].mean()
        return mean_bleu
