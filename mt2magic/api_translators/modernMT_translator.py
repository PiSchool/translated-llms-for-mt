import time

import pandas as pd
from modernmt import ModernMT

from mt2magic.utils.formatting import ModernMT_Parameters

"""
This class is used for calling modernMT API (from Translated) for doing translations
"""
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

    def translate_with_memories(self, data: pd.DataFrame, tmx_reference_file_path: str,
                                dataset_name: str, src_lang: str, trg_lang: str) -> pd.DataFrame:
        """
        Main method to perform translation with modernMT memories. Given a df, reference translations tmx file path
        , source and target lang as input,
        it returns a df with the "translation" column filled.
        Args
            data (:obj:`pd.DataFrame`):     formatting as specified in
                                            mt2magic.formatting.TranslationData
            tmx_reference_file_path (:obj: `str`): The file path of .tmx file for reference translations
            dataset_name (:obj: `str`):     Dataset name
            src_lang (:obj: `str`):         Source language
            trg_lang (:obj: `str`):         Target language


        Returns
            :obj:`pd.DataFrame`:            df with same formatting as the input, containing the "translation"
                                            column filled.
        """

        # As a first step, let's create two memories.
        # One will be used for storing the historical TMX files of the customer (if present) to boost the adaptation.
        tmx_memory = self.mmt.memories.create(f'{dataset_name} {src_lang}-{trg_lang} memory for TMX')
        # The second, will be used to store the corrections for the real-time adaptation feature of ModernMT
        rta_memory = self.mmt.memories.create(f'{dataset_name} {src_lang}-{trg_lang} memory for RTA')

        # Upload the existing TMX file if present
        job = self.mmt.memories.import_tmx(tmx_memory.id, tmx_reference_file_path)
        self.wait_import_job(job)

        source_list = []
        target_list = []
        translation_list = []

        # Now, let's translate the test file and add the corrected sample right-after the translation,
        # simulating the translation job done by a professional translator.
        for i, r in data.iterrows():
            result = self.mmt.translate(src_lang, trg_lang, r['source'], hints=[tmx_memory.id, rta_memory.id])
            job = self.mmt.memories.add(rta_memory.id, src_lang, trg_lang, r['source'], r['target'])
            self.wait_import_job(job)

            source_list.append(r['source'])
            target_list.append(r['target'])
            translation_list.append(result.translation)

        df_result = pd.DataFrame(
            {'source': source_list,
             'target': target_list,
             'translation': translation_list
             })

        return df_result

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

    # Utility method to wait for an import job to be completed
    def wait_import_job(self, import_job, refresh_rate_in_seconds=.25):
        while True:
            time.sleep(refresh_rate_in_seconds)

            import_job = self.mmt.memories.get_import_status(import_job.id)
            if import_job.progress == 1.0:
                break

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

