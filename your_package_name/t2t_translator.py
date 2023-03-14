from translator import Translator
from typing import List


class t2tTranslator(Translator):
    def __init__(self, API_TOKEN: str, API_URL: str):
        super().__init__(API_TOKEN=API_TOKEN, API_URL=API_URL)
        self.src_lan = None
        self.trg_lan = None

    def set_lan(self, src: str, trg: str):
        """
        Updates the inputs language and the target language for translation.
        Args
            src (:obj:`str`): source language (i.e. Italian, Iranian, Spanish...)
            trg (:obj:`str`): target language
        """
        self.src_lan = src
        self.trg_lan = trg

    def translate(self, sentences: List[str]) -> List[str]:
        """
        Translate sentences from the source to the target language; source
        and target language are specified by self.src_lan and self.trg_lan.
        To use custom parameters it's necessary to change self.parameters.
        Args
            sentences (:obj:`list`): sentences in the source language to be translated.
        Returns
            :obj:`list`: list of sentences translated in the target language.
        """
        translations = []
        for sent in sentences:
            translation = self.query(
                {"inputs": f"translate {self.src_lan} to {self.trg_lan}: " + sent,
                "wait_for_model" : True
                 }
            )[0]["translation_text"]
            translations.append(translation)
        return translations
