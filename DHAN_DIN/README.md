# DHAN using DIN sample generation
## prepare data
### method 1
You can get the data from amazon website and process it using the script
```
sh /utils/0_download_raw.sh
```
and then run:
```
python /utils/1_convert_pd.py
python /utils/2_remap_id.py
python /utils/3_build_dataset.py
```
### method 2
Or you can use the file we uploaded, unzip the dataset_electronics.tar.gz .

When you see the files below, you can do the next work.

* dataset.pkl

## train model
Go to the folder has the name of the model you want to run and run following code:
```
python train.py
```
The model blelow had been supported:

* PNN
* Wide (Wide&Deep NN)
* DIN (https://arxiv.org/abs/1706.06978)
* DHAN (Our model)
