# Hakuna-Ma-data
Hakuna Ma-data competition

# Install requirements
Tested on python3.6. Preferable to start with clean virtual env.

`pip install -r requirements.txt`

# Preprocess dataset
I used resized dataset provided by ppleskov
https://community.drivendata.org/t/resized-dataset-is-now-available/3874/21

Download and unzip dataset to a folder put folder name to `DATASET_PATH`. 
For example:

`DATASET_PATH='/home/ivan/projects/datasets/wildlife'`

Expected structure of data:

`ll $DATASET_PATH | grep S | head -3`
- drwxr-xr-x 170 ivan ivan      4096   1 21:43 S1/
- drwxr-xr-x 192 ivan ivan      4096  12 20:28 S10/
- drwxr-xr-x 176 ivan ivan      4096   1 21:42 S2/

`ll $DATASET_PATH"/S10" | grep S | head -3`
- drwxr-xr-x   5 ivan ivan 4096  12 19:48 B08/
- drwxr-xr-x   4 ivan ivan 4096  29 22:34 B09/
- drwxr-xr-x   4 ivan ivan 4096  12 19:48 B010/

`ll  $DATASET_PATH | grep csv`

- -rw-r--r--   1 ivan ivan   3020365   8 20:13 test_metadata.csv
- -rw-r--r--   1 ivan ivan 307504477   8 20:13 train_labels.csv
- -rw-r--r--   1 ivan ivan 495022349   8 20:14 train_metadata.csv

all preprocessed data will be stored to (it is useful if you would like to mound a disk for preprocessed images) 

`mkdir $DATASET_PATH/preprocessed`


DATASET_PATH also contains train_metadata.csv and train_labels.csv.

Open `wildlife/preprocessing/remove_exif.ipynb` and run all. (Takes several hours even on SSD)

Run script to pre-process data. (Takes several hours even on SSD)

`python wildlife/preprocessing/sequences_to_images.py $DATASET_PATH`

It will create `DATASET_PATH/preprocessed/background` and `DATASET_PATH/preprocessed/mean` folders with preprocessed images

You can look at images using wildlife/preprocessing/view.ipynb notebook.

# Train DNN by original images
Run it 2 times to train models for ensemble with arguments:
- path to dataset
- version. if version == 0 it is a fast mode to check if everything works.

For example: 

check if pipeline works:
 
`python wildlife/training/original_images.py $DATASET_PATH 0` 

Run two training pipelines:

`python wildlife/training/original_images.py $DATASET_PATH 1`

`python wildlife/training/original_images.py $DATASET_PATH 2`

One pipeline on a single RTX 2080 Ti + 64 RAM + SSD takes ~ 48 hours.

Pre-trained models can be found by the link https://yadi.sk/d/a1HwAVbvKiIozg. 
The pretrained model were trained slightly different.
I trained on seasons 1 and 2, then downloaded more seasons and fine tuned, then preprocessed images `wildlife/preprocessing/remove_exif.ipynb` and tuned more. 
So the result using this script can be different (better or worse).
(finally I reproduced losses using this script) 

`ll  | grep insres`

- rw-r--r-- 1 ivan ivan 655902600   6 07:47 insres_224_v2.h5
- rw-r--r-- 1 ivan ivan 655902600   6 00:00 insres_299_v0.h5
- rw-r--r-- 1 ivan ivan 655902600   6 17:03 insres_299_v2.h5
- rw-r--r-- 1 ivan ivan 655902600   6 00:03 insres_360_v0.h5
- rw-r--r-- 1 ivan ivan 655902600   6 00:06 insres_384_v0.h5

...


# Train DNN by background images

check mode (fast training):

`python wildlife/training/background_images.py $DATASET_PATH 0`

training mode

`python wildlife/training/background_images.py $DATASET_PATH 1`

`ll  | grep insres  | grep background`

