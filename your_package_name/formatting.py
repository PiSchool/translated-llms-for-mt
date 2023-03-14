from typing import TypedDict, List

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
    score: List[float]

class Parameters(TypedDict):
    """
    Dictionary that contains parameters for generative and t2t models to make API calls.
    For complete documentation:
    https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
    """
    return_full_text: bool
    max_new_tokens: int
    wait_for_model: bool
