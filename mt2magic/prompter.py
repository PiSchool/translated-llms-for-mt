from sentence_transformers import SentenceTransformer
from formatting import PromptConfig
from random import sample

"""
Prompter for pure generative language models.
TODO: prompting by similarity
"""
class Prompter:
    def __init__(self, config: PromptConfig):
        self.n_shot = config["n_shots"]
        self.strategy = config["strategy"]
        self.src_pool = config["src_pool"]
        self.trg_pool = config["trg_pool"]
        if config["model"] is not None:
            self.model = SentenceTransformer(config["model"])
        else:
            self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    def set_config(self, config: PromptConfig):
        self.n_shot = config["n_shots"]
        self.strategy = config["strategy"]
        self.src_pool = config["src_pool"]
        self.trg_pool = config["trg_pool"]
        if config["model"] is not None:
            self.model = SentenceTransformer(config["model"])
        else:
            self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    def get_random_prompt(self, src_sent: str) -> str:
        pool_size = len(self.src_pool)
        idxs = sample(list(range(pool_size)), self.n_shot)
        src_examples = [self.src_pool[idx] for idx in idxs]
        trg_examples = [self.trg_pool[idx] for idx in idxs]
        prompt_sentences = [f"[source]: {src_examples[idx]} [target]: {trg_examples[idx]}"
                            for idx in range(self.n_shot)]
        output = ""
        src_formatting = f"[source]: {src_sent} [target]: "
        for sent in prompt_sentences:
            output += sent
        return output + src_formatting

    def get_prompt(self, src_sent: str) -> str:
        if self.strategy == "random":
            return self.get_random_prompt(src_sent)
        else: # to implement: prompting by similarity
            return src_sent



