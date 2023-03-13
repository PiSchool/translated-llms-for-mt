from transformers import pipeline
from datasets import load_dataset
from datasets import Dataset

import json
import requests
from tqdm.notebook import tqdm as tqdm
import csv

from datetime import datetime
from itertools import islice

# This class contains methods for sending translation requests to the HF API 
# and saving the results to a CSV file
class Translator():

  def __init__(
      self, 
      API_TOKEN, 
      model, 
      path_predictions_folder="../data/interim/predictions_", 
      dataset_location="../data/external/flores200_dataset/devtest/eng_Latn.devtest", 
      dataset_name="Muennighoff/flores200", 
      source_language='eng_Latn', 
      split= "devtest"
  ):
    self.headers = {"Authorization": f"Bearer {API_TOKEN}"}
    self.model = model
    self.API_URL = "https://api-inference.huggingface.co/models/" + model

    self.path_predictions_csv = self.get_valid_csv_name(path_predictions_folder)
    self.source_language = source_language
    self.split = split
    if not dataset_location:
      self.dataset = load_dataset(dataset_name, source_language)[split]
    else:
      self.dataset_location = dataset_location
      self.dataset = self.load_local_dataset()
    self.length_dataset = len(self.dataset)

  # This method generates a valid CSV filename based on the model and the date.
  def get_valid_csv_name(self, path_folder):
    model_cleaned = self.model.split('/', 1)[0]
    csv_name = path_folder + model_cleaned + "_" + datetime.today().strftime('%d-%m-%Y') + ".csv"
    #assert is_valid_filename(csv_name) , "Invalid path to store the csv file, got {}".format(csv_name)
    return csv_name

  # The next two methods are used to load and format a local dataset 
  def format_sentence(self, example):
    return {"sentence": example["text"]}

  def load_local_dataset(self):
    dataset = load_dataset("text", data_files=self.dataset_location)
    formatted_dataset = dataset.map(self.format_sentence)
    return formatted_dataset["train"]

  #This method sends the request for the translation and returns a dictionary
  def query(self, payload):
    data = json.dumps(payload)
    response = requests.request("POST", self.API_URL, 
                                headers=self.headers, data=data)
    return json.loads(response.content.decode("utf-8"))

  # This method retrieves the predictions and saves the results to a CSV file.
  # Optionally, it also returns a list of predictions as a Python list.
  def get_predictions(self, limit=0, return_list_predictions=False):
    row_list = [["source_text", "translated_text",
                 "source_language", "target_language"]] #header for the csv

    if limit==0:
      limit = self.length_dataset

    for _, sample in enumerate(tqdm(islice(self.dataset, limit))):
      sentence = sample["sentence"]
      data = self.query(
          {
              "inputs": sentence,
              "wait_for_model" : True
          }
      )
      try:
        translation = data[0]["translation_text"]
        row_list.append([sentence, translation, "en", "in"])
      except Exception as err:
        print("An exception occurred: {}".format(err))
        continue

    with open(self.path_predictions_csv, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(row_list)

    #return list of predictions without header
    if return_list_predictions: 
      return row_list[1:]