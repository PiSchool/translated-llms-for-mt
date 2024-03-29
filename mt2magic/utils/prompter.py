from sentence_transformers import util, SentenceTransformer
from mt2magic.utils.formatting import PromptConfig
import pandas as pd
from typing import List
import random
from typing import Optional
import torch
random.seed(42)
"""
Prompter for pure generative language models.
"""
class Prompter:
    def __init__(self, config: PromptConfig):
        self.n_shot = config["n_shots"]
        self.strategy = config["strategy"]
        self.src_embeddings = torch.load(config['embeddings_path'])
        self.pool = pd.read_csv(config["pool_path"], sep=config["sep"], encoding=config["encoding"])
        self.embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


    def set_config(self, config: PromptConfig):
        self.n_shot = config["n_shots"]
        self.strategy = config["strategy"]
        self.src_embeddings = torch.load(config['embeddings_path'])
        self.pool = pd.read_csv(config["pool_path"], sep=config["sep"], encoding=config["encoding"])

    def get_random_prompt(self, src_sent: str) -> str:
        """
        Given an input sentence in the source language, it returns a prompt with both
        source examples and the respective translations, chosen randomly in the prompter
        pool. To set the number of shots included in the prompt,set the config.
        Args
            src_sent (:obj:`str`): sentence in the source language to be translated.
        Returns
            (:obj:`str`): A string with the prompt for performing translation.
        """
        pool_size = len(self.pool)
        idxs = random.sample(list(range(pool_size)), self.n_shot)
        return self._format_prompt(src_sent, idxs)

    def get_prompt(self, src_sent: str, label: Optional[str] = None) -> str:
        """
        Given an input sentence in the source language, it returns a prompt with both
        source examples and the respective translations, chosen accordingly to strategy
        (either fuzzy, labeled or random).
        fuzzy:   given an input sentence, the prompt is built with the most similar examples
                 in the pool.
        labeled: given an input sentence and a label, if the dataset has sentences identified
                 by a label it returns a fuzzy prompt performed on the examples with the same
                 label.
        random:  given an input sentence, it returns a prompt with example selected randomly
                 from the pool.
        Args
            src_sent (:obj:`str`): sentence in the source language to be translated.
            label (:obj:`str`):    label of the input sentence.
        Returns
            (:obj:`str`): A string with the prompt for performing translation.
        """
        if self.strategy == "fuzzy":
            return self.get_fuzzy_prompt(src_sent)
        elif self.strategy == "label" and label is not None:
            return self.get_labeled_prompt(src_sent, label)
        else:
            return self.get_random_prompt(src_sent)

    def get_fuzzy_prompt(self, src_sent: str)-> str:
        """
        Given an input sentence in the source language, it returns a prompt with both
        source examples and the respective translations, chosen between the most
        similar ones in the examples pool. To set the number of shots
        included in the prompt, set the config.
        Args
            src_sent (:obj:`str`): sentence in the source language to be translated.
        Returns
            (:obj:`str`): A string with the prompt for performing translation.
        """
        src_sent_embedding = self.embedder.encode(src_sent, convert_to_tensor=True)
        cos_scores = util.cos_sim(src_sent_embedding, self.src_embeddings)[0]
        top_results = torch.topk(cos_scores, k=self.n_shot)[1].tolist()
        return self._format_prompt(src_sent, top_results)

    def get_labeled_prompt(self, src_sent: str, label: str) -> str:
        """
        Given a sentence and its correspondent label, it performs fuzzy prompting on the
        examples of the pool that have the same label as the sentence in input.
        Args
            src_sent (:obj:`str`): sentence in the source language to be translated.
            label (:obj:`str`):    label of the input sentence.
        Returns
            (:obj:`str`): A string with the prompt for performing translation.
        """
        label_idxs = self.pool.loc[self.pool["label"]==label].index.tolist()
        label_embeddings = self.src_embeddings[label_idxs,:]
        src_sent_embedding = self.embedder.encode(src_sent, convert_to_tensor=True)
        label_cos_scores = util.cos_sim(src_sent_embedding, label_embeddings)[0]
        label_top_results = torch.topk(label_cos_scores, k=self.n_shot)[1].tolist()
        top_results = [label_idxs[idx] for idx in label_top_results]
        return self._format_prompt(src_sent, top_results)

    def _format_prompt(self, src_sent: str, idxs: List[int]) -> str:
        """
        Internal method to format the prompt given the source sentence and the
        indexes of the examples to choose from the pool.
        Args
            src_sent (:obj:`str`): sentence in the source language to be translated.
            idxs (:obj:`int`): list of the examples indexes from the pool.
        Returns
            (:obj:`str`): A string with the prompt for performing translation.
        """
        src_examples = [self.pool["source"][idx] for idx in idxs]
        trg_examples = [self.pool["target"][idx] for idx in idxs]
        prompt_sentences = [f"[source]: {src_examples[idx]}\n[target]: {trg_examples[idx]}\n"
                            for idx in range(self.n_shot)]
        output = "Translate the final sentence. Use these translations as an example:\n"
        src_formatting = f"[source]: {src_sent}\n[target]:"
        for sent in prompt_sentences:
            output += sent
        return output + src_formatting

    @staticmethod
    def BLOOM_prompt(type: str, sentence: str, src_lan: str, trg_lan: str):
        """
        Given a sentence to translated from a source language to a target language,
        build a prompt for a BLOOM model using a strategy defined by the following types:
            - 'A': '{src_lan}: {src_sent}
                    {trg_lan}:'
            - 'B': 'Translate the following text from {src_lan} to {trg_lan}:{src_sent}'.
            - 'C': 'Translate the following text from {src_lan} to {trg_lan}:
                    {src_lan}: {src_sent}
                    {trg_lan}:'
        According to out mentor, the first and third methods are better.
        Args:
            type (str) : type of prompting techniques to use. Possible inputs are: 'A', 'B', OR 'C'
            sentence (str) : sentence that we want to translate
            src_lan (str) : source language
            trg_lan (str) : target language
        Returns:
            prompt (str) : prompt to be fed to the model
        """
        prompt = ""
        if type == "A":
            prompt = f"""{src_lan}: {sentence}
            {trg_lan}:"""
        elif type == "B":
            prompt = f"""Translate the following text from {src_lan} to {trg_lan}:{sentence}
            """
        elif type == "C":
            prompt = f"""Translate the following text from {src_lan} to {trg_lan}:
            {src_lan}: {sentence}
            {trg_lan}:"""
        else:
            raise Exception(f"{type} is not a valid type! Valid types are: 'A', 'B', and 'C'.")
        return prompt
