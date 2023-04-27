from mt2magic.outdated.translator import Translator
from typing import List
import pandas as pd


class t2tTranslator(Translator):
    def __init__(self, API_TOKEN: str, API_URL: str):
        super().__init__(API_TOKEN=API_TOKEN, API_URL=API_URL)
        self.src_lan = None
        self.trg_lan = None

    def _set_lan(self, src: str, trg: str):
        """
        Updates the inputs language and the target language for translation.
        Args
            src (:obj:`str`): source language (i.e. Italian, Iranian, Spanish...)
            trg (:obj:`str`): target language
        """
        self.src_lan = src
        self.trg_lan = trg

    def translate_sentences(self, sentences: List[str]) -> List[str]:
        """
        Translate sentences from the source to the target language; source
        and target language are specified by self.src_lan and self.trg_lan.
        Args
            sentences (:obj:`list`): sentences in the source language to be translated.
        Returns
            :obj:`list`: list of sentences translated in the target language.
        """
        translations = []
        for sent in sentences:
            translation = self.query(
                {"inputs": f"translate {self.src_lan} to {self.trg_lan}: " + sent,
                "wait_for_model": True
                 }
            )[0]["translation_text"]
            translations.append(translation)
        return translations

    def translate(self, data: pd.DataFrame, src_lan, trg_lan) -> pd.DataFrame:
        """
        Main method to perform translation. Given a df, source and target
        languages as input, it returns a df with the "translation" column filled.
        Args
            data (:obj:`pd.DataFrame`): formatting as specified in
                                        mt2magic.formatting.TranslationData
            src_lan (:obj:`str`): language of the source sentences (Italian, Iranian, English...)
            trg_lan (:obj:`str`): target language for translation (Spanish, Greek, Dutch...)
        Returns
            :obj:`pd.DataFrame`: df with same formatting as the input, containing the "translation"
                                 column filled.
        """
        source_sentences = data["source"].to_list()
        self._set_lan(src=src_lan, trg=trg_lan)
        data["translation"] = self.translate_sentences(source_sentences)
        return data


