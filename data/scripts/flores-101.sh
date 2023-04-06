# run this script from the path_to/translated-llms-for-mt directory
# as $./data/scripts/flores-101.sh
# here's a link with additional resources: https://github.com/facebookresearch/flores
EXTERNAL_PATH=./data/external
wget -P $EXTERNAL_PATH --trust-server-names https://tinyurl.com/flores200dataset
tar -xf $EXTERNAL_PATH/flores200_dataset.tar.gz -C $EXTERNAL_PATH
rm $EXTERNAL_PATH/flores200_dataset.tar.gz
echo flores-101 dataset in $EXTERNAL_PATH