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

`ll $DATASET_PATH"/S10" | grep S | head -3`
- drwxr-xr-x   5 ivan ivan 4096 дек 12 19:48 B08/
- drwxr-xr-x   4 ivan ivan 4096 ноя 29 22:34 B09/
- drwxr-xr-x   4 ivan ivan 4096 дек 12 19:48 B010/

`ll  $DATASET_PATH | grep csv`

- -rw-r--r--   1 ivan ivan   3020365 окт  8 20:13 test_metadata.csv
- -rw-r--r--   1 ivan ivan 307504477 окт  8 20:13 train_labels.csv
- -rw-r--r--   1 ivan ivan 495022349 окт  8 20:14 train_metadata.csv

DATASET_PATH also contains train_metadata.csv and train_labels.csv.

Open `wildlife/preprocessing/remove_exif.ipynb` and run all. (Takes several hours even on SSD)

Run script to pre-process data. (Takes several hours even on SSD)
`python wildlife/preprocessing/sequences_to_images.py $DATASET_PATH`

It will create `DATASET_PATH/background` and `DATASET_PATH/mean` folders with preprocessed images

You can look at images using wildlife/preprocessing/view.ipynb notebook.

# Train DNN by original images
Run it 2 times to train models for ensemble with arguments:
- path to dataset
- version

Foe example: 

`python wildlife/preprocessing/sequences_to_images.py $DATASET_PATH 1`

`python wildlife/preprocessing/sequences_to_images.py $DATASET_PATH 2`

Pre-trained models can be found by the link https://yadi.sk/d/a1HwAVbvKiIozg. 
The pretrained model were trained slightly different.
I trained on seasons 1 and 2, then downloaded more seasons and fine tuned, then preprocessed images `wildlife/preprocessing/remove_exif.ipynb` and tuned more. 
So the result using this script can be different (better or worse). 
