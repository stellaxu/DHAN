# DHAN using DIEN sample generation
## prepare data
### method 1
You can get the data from amazon website and process it using the script
```
sh /raw_data/0_download_raw.sh
```
and then run:
```
python /DHAN/process_data_electronics.py
python /DHAN/local_aggretor.py
python /DHAN/split_by_user.py
python /DHAN/generate_voc.py
```
### method 2
Or you can use the file we uploaded, unzip the dataset_electronics.tar.gz and copy all the files in the new folder into DHAN folder.
When you see the files below, you can do the next work.

* cat_voc.pkl
* mid_voc.pkl
* uid_voc.pkl
* item-info
* reviews-info
* jointed-new
* jointed-new-split-info
* local_train
* local_test
* local_train_splitByUser
* local_test_splitByUser


## train model
```
python train.py train [model name] 
```
The model blelow had been supported:

* PNN
* Wide (Wide&Deep NN)
* DIN (https://arxiv.org/abs/1706.06978)
* DIEN (https://arxiv.org/abs/1809.03672)
* DHAN (Our model)
