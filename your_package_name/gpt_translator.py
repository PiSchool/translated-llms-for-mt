from translator import Translator
from typing import List
from formatting import Parameters

"""
Pure generative language model class. 
Contains an object of class prompter to get the right prompt for API calls.
Change self.parameters to set parameters for inference:
https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
"""
class gptTranslator(Translator):
    def __init__(self, API_TOKEN: str, API_URL: str, prompter, parameters: Parameters):
        super().__init__(API_TOKEN=API_TOKEN, API_URL=API_URL)
        self.prompter = prompter
        self.parameters = parameters

    def translate(self, sentences: List[str]) -> List[str]:
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
            prompt = self.prompter.get_random_prompt(src=sent)
            output = self.query({'inputs': prompt, 'parameters': self.config})[0]['generated_text']
            translations.append(output)
        return translations



