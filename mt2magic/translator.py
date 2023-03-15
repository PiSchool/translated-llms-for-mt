import json

import pandas as pd
import requests
from typing import List
from abc import abstractmethod, ABC


"""
Abstract base class for the translators (for now adapt to use HF API).
"""
class Translator(ABC):
    def __init__(self, API_TOKEN: str, API_URL: str):
        self.headers = {"Authorization": f"Bearer {API_TOKEN}"}
        self.API_URL = API_URL

    def query(self, payload):
        data = json.dumps(payload)
        response = requests.request("POST", self.API_URL,
                                    headers=self.headers, data=data)
        return json.loads(response.content.decode("utf-8"))
    @abstractmethod
    def translate_sentences(self, sentences: List[str]) -> List[str]:
        pass

    @abstractmethod
    def translate(self, data: pd.DataFrame) -> pd.DataFrame:
        pass




