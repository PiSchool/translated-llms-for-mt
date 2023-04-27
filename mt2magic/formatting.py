from typing import TypedDict, List, Optional, Union

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


class PrompterData(TypedDict):
    """
    pd.DataFrame formatting passed to the Prompter
    """
    source: List[str]
    target: List[str]
    label: Optional[List[str]]


class Parameters(TypedDict):
    """
    Dictionary that contains parameters to make API calls for generative models.
    For complete documentation:
    https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
    """
    return_full_text: bool
    max_new_tokens: int
    wait_for_model: bool


class ModernMT_Parameters(TypedDict):
    """
        Dictionary that contains parameters to make API calls for modernMT.
        For complete documentation:
        https://www.modernmt.com/api/?python
    """
    priority: str
    multiline: bool
    timeout: int
    format: str
    alt_translations: int

class GPT3_Parameters(TypedDict):
    """
        Dictionary that contains parameters to make API calls for open.ai GPT3.
        For complete documentation:
        https://platform.openai.com/docs/api-reference/completions/create
    """
    max_tokens: int
    temperature: float
    stop: Union[str, List]



class PromptConfig(TypedDict):
    """
    Config dict to pass to the prompter to set its state.
    Keys and Values:
        n_shots (:obj:`int`): number of examples to include in the prompt
        strategy (:obj:`str`): strategy to use to perform sampling of the examples
                               (can be either "random", "fuzzy" or "labeled")
        pool_path (:obj:`str`): path to a csv formatted as in mt2magic.formatting.PrompterData
        embeddings_path (:obj:`str`): path to the tensor file (.pt) with the source pool embeddings
        encoding (:obj:`str`): encoding of the pool csv file to be read from pool_path
        sep (:obj:`str`): delimiter used in the pool csv file to be read from pool_path
    """
    n_shots: int
    strategy: str
    pool_path: str
    embeddings_path: str
    encoding:str
    sep: str


