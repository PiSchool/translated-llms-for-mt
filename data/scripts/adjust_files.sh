# run this script from translated-llms-for-mt folder
mkdir ./data/processed/flores
mkdir ./data/processed/translated
mkdir ./data/processed/metrics
mkdir ./data/processed/metrics/flores
mkdir ./data/processed/metrics/translated
mv translated*.* ./data/processed/translated
mv flores*.* ./data/processed/flores
mv en__it ./data/processed/translated/dataset-it-en-big
mv es__it ./data/processed/translated/dataset-it-es-big
