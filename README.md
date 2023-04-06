# Adaptive Machine Translation
Short description of the challenge.

## Directory structure
Update appropriately before handing over this repository. You may want to add other directories/files or remove those you don't need.

```
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   ├── raw            <- The original, immutable data dump
│   └── scripts        <- Scripts to download or generate data
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks. Naming convention is a date (for 
│                         ordering) and a short `_` delimited description, 
│                         e.g. `2022-05-18_initial_data_exploration.ipynb`.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc
│   └── figures        <- Generated graphics and figures
│
├── requirements.txt   <- Required packages (dependencies), e.g. generated 
│                         with `pip freeze > requirements.txt`
│
├── scripts            <- Scripts to train a model, make predictions and so on
│
├── setup.py           <- makes project pip installable (pip install -e .) so 
│                         that `your_package_name` can be imported
└── mt2magic  <- Source code for use in this project
    ├── __init__.py    <- (Optional) Makes `your_package_name` a Python module
    └── *.py           <- Other Python source files (can also be organized in 
                          one or more subdirectories)
```

## How to install
Simple installation from PyPI
```
pip install -r requirements.txt 
```
After installing the necessary libraries, make sure that ```tensorRT``` has been installed successfully by running this command:
```
sudo find / -name libnvinfer*.so* -print
```
You should probably see such an output:
```
Installing collected packages: nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, nvidia-cudnn-cu12, tensorRT
Successfully installed nvidia-cublas-cu12-12.1.0.26 nvidia-cuda-runtime-cu12-12.1.55 nvidia-cudnn-cu12-8.8.1.3 tensorRT-8.6.0
/usr/local/lib/python3.9/dist-packages/tensorrt/libnvinfer_plugin.so.8
/usr/local/lib/python3.9/dist-packages/tensorrt/libnvinfer_builder_resource.so.8.6.0
/usr/local/lib/python3.9/dist-packages/tensorrt/libnvinfer.so.8
```

## Additional data  
First of all, be sure to be placed in the directory ```translated-llms-for-mt```.  
Download the flores dataset by running:  
```
./data/scripts/flores-101.sh
```
Then download the Translated datasets, you can find the cleaned versions on [gDrive](https://drive.google.com/drive/u/4/folders/14E5dAKdK7pwitSqf6zh233YybA73MzvJ): download the files ```translated-it-en-cleaned.csv``` and ```translated-it-es-cleaned.csv``` and put them in ```translated-llms-for-mt```.  
Download all the ```.pt``` files [here](https://drive.google.com/drive/u/4/folders/1qecmn7ySukT6CVZZl2CTPKeN1tq3AHkp) (sBert encodings used for fuzzy prompting) and put them in ```translated-llms-for-mt```.  
Move all the files in the right directories with:  
```
./data/scripts/adjust_files.sh
```
Now launch the two scripts for splitting and formatting the datasets with:
```
python3 -m data.scripts.flores_preprocess
```  
and:  
```
python3 -m data.scripts.translated_split
```  
You can now work with the scripts for evaluation!




## How to run
Experiments with gpt3 and ModernMT (read Additional data paragraph first!)  
You can test ModernMT on four datasets: flores from italian to english (flores_it_en), flores from italian to spanish (flores_it_es), translated from italian to english (translated_it_en), translated from italian to spanish (translated_it_es).  
The inference pipeline exploits Hydra, so adding more experiments is as easy as writing a simple YAML file.  
To run ModernMT evaluation on these datasets, modify the sweeper inputs of ```./configs/config``` like this:
```
defaults:
  - datasets: ???
  - experiments: ???
  - _self_
hydra:
  sweeper:
    params:
      datasets: flores_it_en,flores_it_es,translated_it_en,translated_it_es
      experiments: modernMT
```
Then on the command line run: 
```
python3 -m scripts.hydra_pipeline -m
```
Notice that on the hydra_pipeline.py script there's a "test" boolean flag to perform only the first 5 translations for each dataset.  
You will find the csv with both sentence-wise and aggregate results in ```./data/processed/metrics```.   
Similarly, we can run experiments with gpt3. To perform translations with gpt3 it's necessary to prompt the model, our pipeline provides three strategies to do it: random, fuzzy and label (the latter available only with translated datasets).  
The idea is that we will perform translations by providing few shot examples to the model; the examples are drawn from the development split of the datasets.  
The three strategies differ on the way that examples are picked: "random" means that examples will be chosen randomly from the pool, "fuzzy" instead that the examples will be chosen according to the semantic similarity of the sentences to the source sentence that has to be translated (exploiting cosine similarity and sentence-bert embeddings).  
"label" instead it's a more complex way to choose the example, and to use this strategy we need a pool with sentences labeled by domain (e.g. Translated dataset): the fuzzy matches for prompting are performed only on the sentences belonging to the same group as the source sentence that has to be translated.  
For launching experiments with gpt models, modify the sweeper like this (all the possible configurations are stored in ```./configs/experiments```):  
```
defaults:
  - datasets: ???
  - experiments: ???
  - _self_
hydra:
  sweeper:
    params:
      datasets: flores_it_en,flores_it_es,translated_it_en,translated_it_es
      experiments: gpt_fuzzy_3ex
```
Then launch the script with the same command as the one for modernMT. 


## The team
This challenge, sponsored by [S], was carried out by [X], [Y] and [Z] as part of the [N]th edition of Pi School's School of AI program.
| 1st team member's name  | 2nd team member's name | 3rd team member's name |
| ------------- | ------------- | ------------- |
| ![1st team member](https://cdn.icon-icons.com/icons2/2643/PNG/512/male_man_people_person_avatar_white_tone_icon_159363.png) | ![2nd team member](https://cdn.icon-icons.com/icons2/2643/PNG/512/female_woman_people_person_avatar_black_tone_icon_159371.png) | ![3rd team member](https://cdn.icon-icons.com/icons2/2643/PNG/512/male_man_boy_person_avatar_people_white_tone_icon_159357.png)
| Bio for the 1st team member | Bio for the 2nd team member | Bio for the 3rd team member |
| <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [your_github_handle](https://github.com/your_github_handle)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [your_name_on_linkedin](https://linkedin.com/in/your_linkedin)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) | <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [your_github_handle](https://github.com/your_github_handle)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [your_name_on_linkedin](https://linkedin.com/in/your_linkedin)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) | <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [your_github_handle](https://github.com/your_github_handle)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [your_name_on_linkedin](https://linkedin.com/in/your_linkedin)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) |
