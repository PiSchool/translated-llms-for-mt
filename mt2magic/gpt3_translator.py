import openai
from typing import List

import pandas as pd

from mt2magic.formatting import PromptConfig, GPT3_Parameters
from mt2magic.prompter import Prompter


class gpt3_translator:

    def __init__(self, API_KEY: str, prompt_config: PromptConfig, model_name: str = 'davinci'
                 , param: GPT3_Parameters = None) -> None:
        self.api_key = API_KEY
        self.model_name = model_name
        self.prompter = Prompter(prompt_config)
        self.parameters = param

    def retrieve_model(self, model_name: str):
        """
        Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
        Args
            model_name (:obj:`pd.str`):     The ID of the model to use for this request.

        Returns
            :obj:`open.ai json`: An open.ai json object that contains basic information about the model.
        """
        openai.api_key = self.api_key
        return openai.Model.retrieve(model_name)

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
        openai.api_key = self.api_key
        translations = []
        for sent in sentences:
            prompt = self.prompter.get_prompt(src_sent=sent)
            output = openai.Completion.create(model=self.model_name, prompt=prompt
                                              , suffix=self.parameters['suffix'],
                                              max_tokens=self.parameters['max_tokens']
                                              , temperature=self.parameters['temperature'],
                                              top_p=self.parameters['top_p']
                                              , n=self.parameters['n'], stream=self.parameters['stream']
                                              , logprobs=self.parameters['logprobs'], echo=self.parameters['echo']
                                              , stop=self.parameters['stop'],
                                              presence_penalty=self.parameters['presence_penalty']
                                              , frequency_penalty=self.parameters['frequency_penalty'],
                                              best_of=self.parameters['best_of']
                                              , logit_bias=self.parameters['logit_bias'], user=self.parameters['user'])

            translations.append(output.choices[0]['text'])

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

    def set_model_parameters(self, param: GPT3_Parameters):
        """
        Set model parameters to perform inference.
        For the complete list of parameters cfr your_pacakge_name.formatting.GPT3_Parameters
        Args
            param (:obj:`GPT3_Parameters`): parameters of gpt3 model.
        """
        self.parameters = param