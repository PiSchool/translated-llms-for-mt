from mt2magic.translator import Translator
from typing import List
from mt2magic.formatting import Parameters, PromptConfig
import pandas as pd
from mt2magic.prompter import Prompter

"""
Pure generative language model class. 
gptTranslator contains an object of class Prompter to get the right prompt for API calls.
Change self.parameters to set parameters for inference:
https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
"""
class gptTranslator(Translator):
    def __init__(self, API_TOKEN: str, API_URL: str, prompt_config: PromptConfig, parameters: Parameters):
        super().__init__(API_TOKEN=API_TOKEN, API_URL=API_URL)
        self.prompter = Prompter(prompt_config)
        self.parameters = parameters

    def translate_sentences(self, sentences: List[str]) -> List[str]:
        """
        Translate sentences from the source to the target language; source
        and target language are specified by the prompt.
        To use custom parameters it's necessary to change self.parameters.
        The use of the prompter is WIP.
        Args
            sentences (:obj:`list`): sentences in the source language to be translated.
        Returns
            :obj:`list`: list of sentences translated in the target language.
        """
        translations = []
        for sent in sentences:
            prompt = self.prompter.get_prompt(src_sent=sent)
            output = self.query({'inputs': prompt, 'parameters': self.parameters})[0]['generated_text']
            translations.append(output)
        return translations

    def translate(self, data: pd.DataFrame, prompter_config: PromptConfig) -> pd.DataFrame:
        """
        Main method to perform translation. Given a df as input and a config for the Prompter
        (remember that target language is specified by the prompt in generative models),
        it returns a df with the "translation" column filled.
        To use custom parameters for the generative model it's necessary to change self.parameters
        before this call with self.set_model_parameters
        Args
            data (:obj:`pd.DataFrame`):             formatting as specified in
                                                    mt2magic.formatting.TranslationData

            prompter_config (:obj: `PromptConfig`): formatting as specified in
                                                    mt2magic.formatting.PromptConfig
        Returns
            :obj:`pd.DataFrame`: df with same formatting as the input, containing the "translation"
                                 column filled.
        """
        self._set_prompter_config(config=prompter_config)
        source_sentences = data["source"].to_list()
        data["translation"] = self.translate_sentences(source_sentences)
        return data

    def _set_prompter_config(self, config: PromptConfig):
        """
        Set prompt config.
        cfr your_pacakge_name.formatting.PromptConfig for more info.
        Args
            config (:obj:`PromptConfig`): config for the Prompter.
        """
        self.prompter.set_config(config)

    def set_model_parameters(self, param: Parameters):
        """
        Set model parameters to perform inference.
        For the complete list of parameters cfr your_pacakge_name.formatting.Parameters
        Args
            param (:obj:`Parameters`): parameters of gpt model.
        """
        self.parameters = param



