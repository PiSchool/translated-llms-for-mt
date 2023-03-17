# from mt2magic.prompting import get_random_prompt
# from mt2magic.dataset_utils import get_eval_sentences

from datasets import load_dataset

import json
import requests
from tqdm.notebook import tqdm as tqdm
import csv
import os

from random import sample
# This class contains methods for sending translation requests to the HF API 
# and saving the results to a CSV file
class Translator():

  def __init__(
      self, 
      API_TOKEN, 
      path_predictions_folder="data/interim/", 
      dataset_location="data/external/flores200_dataset/", 
      dataset_name="Muennighoff/flores200", 
      split= "devtest"
  ):
    
    self.headers = {"Authorization": f"Bearer {API_TOKEN}"}
    self.path_predictions_folder = path_predictions_folder
    self.split = split
    if not dataset_location:
      self.dataset = load_dataset(dataset_name)[split]
    else:
      self.dataset_location = dataset_location

  def get_random_prompt(self, src_lan: str, trg_lan: str, src: str, n: int = 5) -> str:
    """
    :param src_lan: name of the file which contains source languages examples
    :param trg_lan: name of the file which contains target languages examples
    :param src: sentence to translate
    :param n: number of examples to include in the prompt
    :return: prompt string
    """
    with open(src_lan) as f:
        examples = f.readlines()
        size = len(examples)
        idxs = sample(list(range(size)), n)
        src_examples = [examples[idx] for idx in idxs]
    with open(trg_lan) as f:
        examples = f.readlines()
        trans_examples = [examples[idx] for idx in idxs]
    prompt_sentences = [f"[source]: {src_examples[idx]} [target]: {trans_examples[idx]}"
                        for idx in range(n)]
    output = ""
    src_formatting = f"[source]: {src} [target]:"
    for sent in prompt_sentences:
        output += sent
    return output + src_formatting


  def get_eval_sentences(self, src_lan: str, trg_lan: str, num_samples: int=0):
    output = {'source': [], 'target': []}
    with open(src_lan) as f:
        examples = f.readlines()
        if num_samples != 0:
            output['source'] = examples[:num_samples]
        else:
            output['source'] = examples
    with open(trg_lan) as f:
        examples = f.readlines()
        if num_samples != 0:
            output['target'] = examples[:num_samples]
        else:
            output['target'] = examples
    output['translation'] = []
    return output
  
  # This method generates a valid CSV filename based on the model and the date.
  def get_valid_csv_path(self, model_name, src_lang, trg_lang):
    csv_name = self.path_predictions_folder + model_name + "_" + src_lang + "_" + trg_lang + ".csv"
    #assert is_valid_filename(csv_name) , "Invalid path to store the csv file, got {}".format(csv_name)
    return csv_name

  # The next two methods are used to load and format a local dataset 
  def format_sentence(self, example):
    return {"sentence": example["text"]}

  def load_local_dataset(self, split:str):
    dataset_path = self.dataset_location + split
    dataset = load_dataset("text", data_dir=dataset_path)
    formatted_dataset = dataset.map(self.format_sentence)
    return formatted_dataset["train"]

  # this method may be not necessary anymore if the model changes
  def extract_string_difference(self, big_str: str, small_str: str) -> str:
      small_str_len = len(small_str)
      return big_str[small_str_len:]

  #This method sends the request for the translation and returns a dictionary
  def query(self, payload):
    data = json.dumps(payload)
    response = requests.request("POST", self.API_URL, 
                                headers=self.headers, data=data)
    return json.loads(response.content.decode("utf-8"))
        
  # This method retrieves the predictions and saves the results to a CSV file.
  # Optionally, it also returns a list of predictions as a Python list.
  def get_predictions(
        self, src_language="data/external/flores200_dataset/devtest/eng_Latn.devtest", 
        trg_language="data/external/flores200_dataset/devtest/ita_Latn.devtest", 
        limit=1, 
        modeltype="Helsinki", 
        return_list_predictions=False
        ):

    if modeltype == "t5":
      model = "t5-base"
    elif modeltype == "gpt":
      model = "gpt2"#"https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
    else:
      modeltype = "helsinki"
      model = "Helsinki-NLP/opus-mt-en-it"
    self.API_URL = "https://api-inference.huggingface.co/models/" + model

    row_list = [["source", "target", "translation"]] #header for the csv

    trans_data = self.get_eval_sentences(src_lan=src_language, trg_lan=trg_language, num_samples=limit)
    print("Translating with {} model...".format(model))

    for src_sent, trg_sent in zip(trans_data['source'], trans_data['target']):
      if modeltype == "t5":
        data = self.query(
            {
                "inputs": "translate English to French: " + src_sent,
                "wait_for_model" : True
            }
        )
      elif modeltype == "gpt":
        src_example = src_language.replace("devtest", "dev" )
        trg_example = trg_language.replace("devtest", "dev" )
        prompt = self.get_random_prompt(src_lan=src_example, trg_lan=trg_example, src=src_sent, n=1)
        data = self.query({'inputs': prompt, 'temperature': 100, 'top_k': 2, 'return_full_text': True,
                            'num_return_sequences': 3})#[0]['generated_text']
      else:#Helsinki
        data = self.query(
            {
                "inputs": src_sent,
                "wait_for_model" : True
            }
        )
      try:
        if modeltype == "gpt":
          generated_text = data[0]['generated_text']
          translation = self.extract_string_difference(generated_text, prompt)
        else:
          translation = data[0]["translation_text"]
        row_list.append([src_sent, trg_sent, translation])
      except Exception as err:
        print("An exception occurred: {}".format(err))
        continue

    # Time to save results 
    src_file = os.path.splitext(os.path.basename(src_language))[0]
    trg_file = os.path.splitext(os.path.basename(trg_language))[0]
    path_predictions_csv = self.get_valid_csv_path(modeltype, src_file, trg_file)
    with open(path_predictions_csv, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(row_list)

    print("Done!")

    #return list of predictions without header
    if return_list_predictions: 
      return row_list[1:]
    

  def get_predictions_every_language(self, modeltype="Helsinki", limit=1):
    dir_eval = self.dataset_location + "devtest/"
    encoding = 'utf-8'
    directory = os.fsencode(dir_eval)
    print(directory)
    for source in os.listdir(directory):
        filesource = os.fsdecode(source)
        for target in os.listdir(directory):
          filetarget = os.fsdecode(target)
          if filesource != filetarget:
            src_l = str(directory, encoding) + filesource
            trg_l = str(directory, encoding) + filetarget
            self.get_predictions(modeltype=modeltype, limit=limit, src_language=src_l, trg_language=trg_l)
    