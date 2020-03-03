# Hakuna-Ma-data
Hakuna Ma-data competition

# Install requirements
Tested on python3.6. Preferable to start with clean virtual env.

`pip install -r requirements.txt`

# Preprocess dataset
I used resized dataset provided by ppleskov
https://community.drivendata.org/t/resized-dataset-is-now-available/3874/21

Download and unzip dataset to a folder put folder name to `DATASET_PATH`. For example:
`DATASET_PATH='/home/ivan/projects/datasets/wildlife'`
Expected structure of data:

`ll $DATASET_PATH | grep S | head -3`
- drwxr-xr-x 170 ivan ivan      4096 дек  1 21:43 S1/
- drwxr-xr-x 192 ivan ivan      4096 дек 12 20:28 S10/
- drwxr-xr-x 176 ivan ivan      4096 дек  1 21:42 S2/

`ll $DATASET_PATH"/S10" | head -3`
- drwxr-xr-x   5 ivan ivan 4096 дек 12 19:48 B03/
- drwxr-xr-x   4 ivan ivan 4096 ноя 29 22:34 B04/
- drwxr-xr-x   4 ivan ivan 4096 дек 12 19:48 B05/

DATASET_PATH also contains train_metadata.csv and train_labels.csv.

Run script to pre-process data. (Takes several hours even on SSD)
`python wildlife/preprocessing/sequences_to_images.py $DATASET_PATH`

It will create `DATASET_PATH/background` and `DATASET_PATH/mean` folders with preprocessed images

You can look at images using wildlife/preprocessing/view.ipynb notebook.




