from typing import TypedDict, List, Union

"""
File that contains structures of the types we're using.
"""

class TranslationData(TypedDict):
    """
    Agreed data formatting for csv files.
    Every pd.DataFrame should have the same columns as this formatting.
    """
    source: List[str]
    target: List[str]
    translation: List[str]
    BLEU: List[float]
    chrf: List[float]
    COMET: List[float]

class Parameters(TypedDict):
    """
    Dictionary that contains parameters to make API calls for generative models.
    For complete documentation:
    https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
    """
    return_full_text: bool
    max_new_tokens: int
    wait_for_model: bool

class PromptConfig(TypedDict):
    """
    Config dict to pass to the prompter to set its state.
    Keys and Values:
        n_shots (:obj:`int`): number of examples to include in the prompt
        strategy (:obj:`str`): strategy to use to perform sampling of the examples
                               (can be either "random" or "similar")
        src_pool (:obj:`list`): list of examples for the source langauge
        trg_pool (:obj:`list`): list of examples for the target langauge
        model (:obj:`str`): SentenceTransformer model to use if strategy is set to "similar"
    """
    n_shots: int
    strategy: str
    src_pool: List[str]
    trg_pool: List[str]
    model: Union[str, None]

