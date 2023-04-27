from typing import List
import tiktoken

"""
This class is used for mainly 2 things: 
1) Get an estimation of costs of fine-tuning and inference on different datasets, 
   using scripts/cost_estimation_openai.py
2) Spot the sentences that are too long to be handled by the context-length of 
   openai models.   
"""
class openaiTokenizer:
    def __init__(self):
        # cost for gpt-4 is 0.06 in completion and 0.03 for prompts
        self.inference_cost_per_token = {'davinci': 0.02/1000, 'gpt-3.5-turbo': 0.002/1000,
                                    'gpt-4': 0.03/1000}
        # cost for davinci fine-tuning is 0.03, but using it costs 0.12
        self.fine_tune_cost_per_token = 0.03/1000
    @staticmethod
    def num_tokens_from_string(sentence: str, model_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(sentence))
        return num_tokens

    def translation_cost(self, sources: List[str], prompts: List[str], model_name: str) -> float:
        """
        Given a list of source sentences, their correspondent prompts and the model name,
        return an estimate of the cost of performing the translation with the assumption
        that the translated sentence will have the same length of the input sentence.
        Args:
            sources (:obj:`list`): list of the source sentences to be translated
            prompts (:obj:`list`): list of the prompts used to translate the source sentences
            model_name (:obj:`str`): name of the model used to perform the translations
        Returns
            :obj:`float`: estimate of the cost of translating all the input sentences.
        """
        cost = 0
        cost_per_token = self.inference_cost_per_token[model_name]
        # cost for gpt-4 is 0.06 in completion and 0.03 for prompts
        multiplier = 2 if model_name == 'gpt-4' else 1
        for source, prompt in zip(sources, prompts):
            cost += self.num_tokens_from_string(prompt, model_name) * cost_per_token
            cost += self.num_tokens_from_string(source, model_name) * cost_per_token * multiplier
        return cost

    def finetune_cost(self, sources: List[str], prompts: List[str]) -> float:
        """
        Given a list of source sentences, their correspondent prompts and the model name,
        return an estimate of the cost of performing the translation with the assumption
        that the translated sentence will have the same length of the input sentence.
        Args:
            sources (:obj:`list`): list of the source sentences to be translated
            prompts (:obj:`list`): list of the prompts used to translate the source sentences
        Returns
            :obj:`float`: estimate of the cost of translating all the input sentences.
        """
        cost = 0
        cost_per_token = 0.03/1000
        for source, prompt in zip(sources, prompts):
            cost += self.num_tokens_from_string(prompt, 'davinci') * cost_per_token
            cost += self.num_tokens_from_string(source, 'davinci') * cost_per_token
        return cost

    def finetuned_translation_cost(self, sources: List[str], prompts: List[str]) -> float:
        """
        This method is used for getting the translation costs using a fine-tuned gpt-3.
        The assumption is that the translated sentences have the same length of the source
        sentences.
        """
        return self.translation_cost(sources, prompts, 'davinci') * 6







