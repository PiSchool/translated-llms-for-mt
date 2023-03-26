import pandas as pd
from modernmt import ModernMT


class modernMT_translator(object):
    def __init__(self, api_key, source_lang, target_lang, platform=None, platform_version=None) -> None:
        self.mmt = ModernMT(api_key, platform, platform_version)
        self.source_lang = source_lang
        self.target_lang = target_lang

    def list_supported_languages(self):
        """
        Gives the list of supported language codes (ISO 639-1) for translation.
        Args
            None
        Returns
            :obj:`List`: Returns an array which contains supported language codes.

        """
        return self.mmt.list_supported_languages()

    def translate_sentences(self, q, context_vector=None, hints=None, options= None):
        """
        Allows to translate an input text or a list of texts.
        Args
            q (:obj:`str` or `List`): Both string (single sentence) and list (list of sentences) can be passed.

            context_vector (:obj: `str`): Contains a list of couples separated by a comma.
                                          Each couple contains the memory id and
                                          the corresponding weight separated by a colon.

            hints (:obj: `str`): Contains a list of memory ids separated by a comma.
                                 It can be used to force the assignment of the maximum weight
                                  to the given list of memories.

        Returns
            :obj: `str` or `List`: Returns a str if a single sentence is passed
                                   and return a list of translations if a list of sentences is passed.

        """

        translation = self.mmt.translate(self.source_lang, self.target_lang, q
                                         , context_vector=context_vector, hints=hints, options=options)
        if type(translation) is list:
            translation_list = []
            for trans in translation:
                translation_list.append(trans.translation)
            return translation_list
        else:
            return translation.translation

    def translate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to perform translation. Given a df as input,
        it returns a df with the "translation" column filled.
        Args
            data (:obj:`pd.DataFrame`):     formatting as specified in
                                            mt2magic.formatting.TranslationData

        Returns
            :obj:`pd.DataFrame`:            df with same formatting as the input, containing the "translation"
                                            column filled.
        """
        source_sentences = data["source"].to_list()
        data["translation"] = self.translate(source_sentences)
        return data

    def set_languages(self, source_lang: str, target_lang: str) -> None:
        """
        Set source and target languages.
        Args
            source_lang (:obj:`str`): source language.
            target_lang (:obj:`str`): target language.
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
