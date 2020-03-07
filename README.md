# Install requirements
Tested on python3.6. Preferable to start with clean virtual env.

`pip install -r requirements.txt`

# Preprocess dataset
I used resized dataset provided by ppleskov

https://community.drivendata.org/t/resized-dataset-is-now-available/3874/21

Setup vaiable `DATASET_PATH`

For example:

`DATASET_PATH='/home/ivan/projects/datasets/wildlife'`

Download and unzip dataset to a `DATASET_PATH` folder. 

Expected structure of data:

`ll $DATASET_PATH | grep S | head -3`
- drwxr-xr-x 170 ivan ivan      4096   1 21:43 S1/
- drwxr-xr-x 192 ivan ivan      4096  12 20:28 S10/
- drwxr-xr-x 176 ivan ivan      4096   1 21:42 S2/

`ll $DATASET_PATH"/S10" | grep S | head -3`
- drwxr-xr-x   5 ivan ivan 4096  12 19:48 B08/
- drwxr-xr-x   4 ivan ivan 4096  29 22:34 B09/
- drwxr-xr-x   4 ivan ivan 4096  12 19:48 B010/

`DATASET_PATH` also contains train_metadata.csv and train_labels.csv.


`ll  $DATASET_PATH | grep csv`

- -rw-r--r--   1 ivan ivan   3020365   8 20:13 test_metadata.csv
- -rw-r--r--   1 ivan ivan 307504477   8 20:13 train_labels.csv
- -rw-r--r--   1 ivan ivan 495022349   8 20:14 train_metadata.csv

All preprocessed data will be stored to:

`$DATASET_PATH/preprocessed`

Create the dir:

`mkdir $DATASET_PATH/preprocessed`

Open https://github.com/BraginIvan/Hakuna-Ma-data/blob/master/wildlife/preprocessing/remove_exif.ipynb and run all. (Takes several hours on SSD)

Run script to create background and mean images/.

`python wildlife/preprocessing/sequences_to_images.py $DATASET_PATH`

It will create `DATASET_PATH/preprocessed/background` and `DATASET_PATH/preprocessed/mean` folders with preprocessed images

You can look at images using wildlife/preprocessing/view.ipynb notebook.


# Train DNN by original images


arguments:
- path to dataset ($DATASET_PATH)
- mode (0 - fast mode to be sure everything works) (1 and 2 slightly different pipelines foe ensembling)

check if pipeline works:
 
`python wildlife/training/original_images.py $DATASET_PATH 0` 


`ll  | grep insres`

- rw-r--r-- 1 ivan ivan 655902600   6 07:47 insres_224_v0.h5
- rw-r--r-- 1 ivan ivan 655902600   6 00:00 insres_299_v0.h5
- rw-r--r-- 1 ivan ivan 655902600   6 00:03 insres_360_v0.h5
- rw-r--r-- 1 ivan ivan 655902600   6 00:06 insres_384_v0.h5

Run two training pipelines:

`python wildlife/training/original_images.py $DATASET_PATH 1`

`python wildlife/training/original_images.py $DATASET_PATH 2`

One pipeline on a single RTX 2080 Ti + 64 RAM + SSD takes ~ 48 hours.

Pre-trained models can be found by the link https://yadi.sk/d/a1HwAVbvKiIozg. 


# Train DNN by background images

arguments:
- path to dataset ($DATASET_PATH)
- mode (0 - fast mode to be sure everything works) (1 training mode)


check mode (fast training):

`python wildlife/training/background_images.py $DATASET_PATH 0`

`ll  | grep insres  | grep background`


training mode

`python wildlife/training/background_images.py $DATASET_PATH 1`



# Train DNN by concatination of background and mean images

arguments:
- path to dataset ($DATASET_PATH)
- mode (0 - fast mode to be sure everything works) (1 training mode)


check mode (fast training):

`python wildlife/training/concat_back_mean.py $DATASET_PATH 0`

`ll  | grep insres  | grep concat`


training mode

`python wildlife/training/concat_back_mean.py $DATASET_PATH 1`


# prepare csv with predictions of seasons 10 and 9 for boosting

# train boosting


