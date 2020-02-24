#DHAN using DIEN sample generation
#prepare data
You can get the data from amazon website and process it using the script
sh /raw_data/0_download_raw.sh

When you see the files below, you can do the next work.

cat_voc.pkl
mid_voc.pkl
uid_voc.pkl
local_train_splitByUser
local_test_splitByUser
train model
python train.py train [model name] 
The model blelow had been supported:

PNN
Wide (Wide&Deep NN)
DIN (https://arxiv.org/abs/1706.06978)
DIEN (Our model)
DHAN (Our model)
