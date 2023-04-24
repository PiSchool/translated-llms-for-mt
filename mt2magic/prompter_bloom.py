def BLOOM_prompt(type: str, sentence: str, src_lan: str, trg_lan:str):
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




