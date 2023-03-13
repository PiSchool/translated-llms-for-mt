
from translator_api import Translator

def main():

    print("Enter the API Token for HuggingFace: ")
    API_TOKEN = input()
    model = "Helsinki-NLP/opus-mt-en-it"

    translator = Translator(API_TOKEN, model)
    print("Translating...\n")
    l = translator.get_predictions(limit=10)
    print("\nDone!")

if __name__=="__main__":
    main()