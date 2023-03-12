import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize


class Translator():

    def __init__(
            self,
            source_lang,
            source_lang_filePath,
            target_lang,
            target_lang_filePath
    ):
        self.source_lang = source_lang
        self.target_lang_filePath = source_lang_filePath

        self.target_lang = target_lang
        self.target_lang_filePath = target_lang_filePath

    def create_reference_dataset(self):
        try:
            with open(self.target_lang_filePath, 'r', encoding='utf-8') as source_lang_data:
                source_data_list = [line for line in source_lang_data]
        except (Exception,):
            print('The first file can not be read!')

        try:
            with open(self.target_lang_filePath, 'r', encoding='utf-8') as target_lang_data:
                target_data_list = [line for line in target_lang_data]
        except (Exception,):
            print('The second file can not be read!')

        df_reference = pd.DataFrame({self.source_lang: source_data_list, self.target_lang: target_data_list})
        return df_reference

    def calculate_bleu_score(self, df_translation):
        cnt = 0
        bleu_score = 0
        df_reference = self.create_reference_dataset()
        for i, r in df_translation.iterrows():
            source_lang = r['source_language']
            target_lang = r['target_language']
            if self.source_lang == source_lang & self.target_lang:
                source_text = r['source_text']
                translated_text = r['translated_text']
                search_result = df_reference.loc[df_reference['source'] == source_text]
                if len(search_result) == 1:
                    reference_target_text = search_result['target'][0]
                    bleu_score += sentence_bleu(word_tokenize(reference_target_text), word_tokenize(translated_text))
                    cnt += 1
                if len(search_result) > 1:
                    reference_target_text_list = []
                    for text in search_result['target'].values:
                        token_list = word_tokenize(text)
                        reference_target_text_list.append(token_list)
                        bleu_score += corpus_bleu([reference_target_text_list], [word_tokenize(translated_text)])
                        cnt += 1
            else:
                print(
                    f'The BLEU score can not be calculated for index {i}, source and target lang is different from the reference!')

        if cnt != 0:
            final_bleu_score = bleu_score / cnt
            return final_bleu_score
        else:
            print('The BLEU score could not be calculated!')
            return None
