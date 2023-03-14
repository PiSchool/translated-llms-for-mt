from translator_api import Translator
from Evaluator import Evaluator

import warnings
warnings.filterwarnings("ignore")

# src_ex = "data/external/flores200_dataset/dev/eng_Latn.dev"
# trg_ex = "data/external/flores200_dataset/dev/ita_Latn.dev"
# src_eval = "data/external/flores200_dataset/devtest/eng_Latn.devtest"
# trg_eval = "data/external/flores200_dataset/devtest/fra_Latn.devtest"

def main():

    print("Enter the API TOKEN: ")
    API_TOKEN = "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"#input() 

    print("Enter which model you want to use (t5, Helsinki, gpt): ")
    model= input()

    translator = Translator(API_TOKEN)
    
    #translator.get_predictions(src_eval, trg_eval, modeltype=model, limit=20)

    #translator.get_predictions_every_language(model, limit=5)
    translator.get_predictions(modeltype=model, limit=10)

    eval = Evaluator()
    df_translation = eval.bleu_score_from_file_path('data/interim/Helsinki_eng_Latn_ita_Latn.csv')
    
    result = eval.calculate_corpus_bleu(df_translation)
    print("The BLEU score is: {}".format(result))

if __name__=="__main__":
    main()