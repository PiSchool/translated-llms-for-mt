from translator import Translator
from typing import List


class mtTranslator(Translator):
    def __init__(self, API_TOKEN: str, API_URL: str):
        super().__init__(API_TOKEN=API_TOKEN, API_URL=API_URL)

    def set_lan(self, src: str, trg: str):
        """
        Updates the inputs language and the target language for translation
        by updating the model used for inference
        (complete list of models available: https://huggingface.co/Helsinki-NLP)
        Args
            src (:obj:`str`): source language (i.e. it, en, es...)
            trg (:obj:`str`): target language
        """
        self.API_URL = "https://api-inference.huggingface.co/models/" \
                        + f"Helsinki-NLP/opus-mt-{self.src}-{self.trg}"

    def translate(self, sentences: List[str]) -> List[str]:
        """
        Translate sentences from the source to the target language; source
        and target language are specified by the model used.
        Args
            sentences (:obj:`list`): sentences in the source language to be translated.
        Returns
            :obj:`list`: list of sentences translated in the target language.
        """
        translations = []
        for sent in sentences:
            translation = self.query({"inputs": sent, "wait_for_model": True})[0]["translation_text"]
            translations.append(translation)
        return translations



