import pandas as pd
from modernmt import ModernMT

from mt2magic.formatting import ModernMT_Parameters


class modernMT_translator(object):
    def __init__(self, api_key, source_lang, target_lang, platform=None, platform_version=None
                 , param: ModernMT_Parameters = None) -> None:
        self.mmt = ModernMT(api_key, platform, platform_version)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.parameters = param

    def list_supported_languages(self):
        """
        Gives the list of supported language codes (ISO 639-1) for translation.
        Args
            None
        Returns
            :obj:`List`: Returns an array which contains supported language codes.

        """
        return self.mmt.list_supported_languages()

    def translate_sentences(self, q, context_vector=None, hints=None):
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
            :obj: `List`: Returns a list of translations.

        """
        # ModernMT model just accept a maximum of 128 sentences at each API call
        batch_size = 128
        data_batches = [q[i:i + batch_size] for i in range(0, len(q), batch_size)]
        translation_list = []
        for batch in data_batches:
            translation = self.mmt.translate(self.source_lang, self.target_lang, batch
                                         , context_vector=context_vector, hints=hints, options=self.parameters)
            if type(translation) is list:
                for trans in translation:
                    translation_list.append(trans.translation)
            else:
                translation_list.append(translation.translation)

        return translation_list

    def translate(self, data: pd.DataFrame, context_vector=None, hints=None) -> pd.DataFrame:
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
        data["translation"] = self.translate_sentences(source_sentences, context_vector, hints)
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

    def set_model_parameters(self, param: ModernMT_Parameters):
        """
        Set model parameters to perform inference.
        For the complete list of parameters cfr your_pacakge_name.formatting.ModernMT_Parameters
        Args
            param (:obj:`ModernMT_Parameters`): parameters of modernMT model.
        """
        self.parameters = param

