
from translator_api import Translator

src_ex = "data/external/flores200_dataset/dev/eng_Latn.dev"
trg_ex = "data/external/flores200_dataset/dev/ita_Latn.dev"
src_eval = "data/external/flores200_dataset/devtest/eng_Latn.devtest"
trg_eval = "data/external/flores200_dataset/devtest/fra_Latn.devtest"

def main():

    print("Enter which model you want to use (t5, Helsinki, gpt): ")
    API_TOKEN = "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"#input() 

    model= input()

    translator = Translator(API_TOKEN)
    
    #translator.get_predictions(src_eval, trg_eval, modeltype=model, limit=20)

    translator.get_predictions_every_language(model, limit=5)

if __name__=="__main__":
    main()