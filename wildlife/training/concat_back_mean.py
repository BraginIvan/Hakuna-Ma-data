import json
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# from IPython.display import Image
from tensorflow.keras.metrics import categorical_accuracy
import cv2


pd.set_option('max_colwidth', 80)

# This is where our downloaded images and metadata live locally
DATA_PATH = Path("/home/ivan/projects/datasets/wildlife")

MEAN_DATASET_PATH = Path("/media/ivan/data/mean")
BACK_DATASET_PATH = Path("/home/ivan/projects/datasets/wildlife/background")

train_metadata = pd.read_csv(DATA_PATH / "train_metadata.csv")
train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", index_col="seq_id")

train_metadata['season'] = train_metadata.seq_id.map(lambda x: x.split('#')[0])
train_metadata = train_metadata.sort_values('file_name').set_index('seq_id')

train_labels = train_labels[train_labels.index.isin(train_metadata.index)]

count = train_metadata.groupby('seq_id').size().reset_index().set_index('seq_id')
count.columns = ['seq_id_count']
train_metadata = train_metadata.join(count, on='seq_id', how='left')

train_metadata = train_metadata[train_metadata.seq_id_count > 1]
train_metadata=train_metadata.sort_values('file_name').groupby('seq_id').first()

train_metadata['season'] = train_metadata.index.map(lambda x: x.split('#')[0])
train_metadata['cam_id'] = train_metadata.index.map(lambda x: x.split('#')[1])
train_metadata['angle_id'] = train_metadata.index.map(lambda x: x.split('#')[2])

train_seasons = ['SER_S1','SER_S2', 'SER_S3', 'SER_S4', 'SER_S5', 'SER_S6', 'SER_S7', 'SER_S8']
val_seasons = ["SER_S9"]

val_x = train_metadata[train_metadata.season.isin(val_seasons)]
val_y = train_labels[train_labels.index.isin(val_x.index)]

train_metadata = train_metadata[train_metadata.season.isin(train_seasons)]
train_labels = train_labels[train_labels.index.isin(train_metadata.index)]

train_gen_df = train_labels.join(train_metadata)
val_gen_df = val_y.join(val_x)
label_columns = train_labels.columns.tolist()


val_gen_df['mean_file_name'] = val_gen_df.apply(
    lambda x: str(MEAN_DATASET_PATH) + "/"+ 'val/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)


val_gen_df['back_file_name'] = val_gen_df.apply(
    lambda x: str(BACK_DATASET_PATH) + "/"+ 'background/val/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

train_gen_df['mean_file_name'] = train_gen_df.apply(
    lambda x: str(MEAN_DATASET_PATH) + "/"+ 'train/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

train_gen_df['back_file_name'] = train_gen_df.apply(
    lambda x: str(BACK_DATASET_PATH) + "/"+ 'background/train/' + ('empty' if x['empty'] ==1 else 'animal') + "/"+ x.season+ "/"+ x.cam_id+ "/"+x.angle_id+ "/"+ x.file_name.split('/')[-1], axis=1
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# target_size = (360, 480)
target_size = (384, 512)
target_size2= (512, 384)
batch_size = 16

from tensorflow.keras.applications import inception_resnet_v2

# Note that we pass the preprocessing function here
datagen = ImageDataGenerator()

train_datagen_x = datagen.flow_from_dataframe(
    dataframe=train_gen_df,
    x_col="mean_file_name",
    y_col=label_columns,
    class_mode="other",
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True
)
val_datagen_x = datagen.flow_from_dataframe(
    dataframe=val_gen_df.iloc[::20],
    x_col="mean_file_name",
    y_col=label_columns,
    class_mode="other",
    target_size=target_size,
    batch_size=batch_size,
    shuffle=True
)

import random
def back_mean_gen(df, flip=True):
    while True:
        sample = df.sample(n = 8)
        mean_imgs = np.array([inception_resnet_v2.preprocess_input(cv2.resize(cv2.imread(p), target_size2)[:,:,::-1]) for p in sample.mean_file_name])
        back_imgs = np.array([cv2.resize(cv2.imread(p), target_size2)[:,:,::-1]/255 for p in sample.back_file_name])
        if flip and random.choice([True, False]):
            mean_imgs=mean_imgs[:,:,::-1,:]
            back_imgs=back_imgs[:,:,::-1,:]
        labels = sample[label_columns].values
        yield ((back_imgs, mean_imgs), labels)

val_datagen = back_mean_gen(val_gen_df[val_gen_df.mean_file_name.isin(val_datagen_x.filenames)], flip=False)
train_datagen = back_mean_gen(train_gen_df[train_gen_df.mean_file_name.isin(train_datagen_x.filenames)])

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D, Input, Lambda,Embedding,Concatenate,Flatten, Conv2D

from tensorflow.keras.models import load_model
import tensorflow as tf
model_mean = load_model("./eval_26_11/mean_insres_all_lr.h5")
model_back = load_model("./eval_26_11/background_insres_all_lr.h5")
for layer in model_mean.layers:
    layer._name = str('mean_') + layer.name
for layer in model_back.layers:
    layer._name = str('back_') + layer.name


back_input = model_back.layers[-4].output
mean_input = model_mean.layers[-4].output

x = Concatenate(axis=3)([back_input, mean_input])
x = Conv2D(filters=1024, kernel_size=3, padding='valid')(x)
x = GlobalMaxPooling2D()(x)
x = Dense(54, activation="sigmoid")(x)
model = Model(inputs=[model_back.input, model_mean.input], outputs=x)
model.summary()



model.save("connected_model_v2.h5")



from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import categorical_accuracy

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-1000:]:
    layer.trainable = True


rms = Adam(learning_rate=0.0003)

model.compile(optimizer=rms, loss="binary_crossentropy", metrics=[categorical_accuracy])

from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.metrics import  categorical_accuracy


for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[categorical_accuracy])

def scheduler(epoch):
    lr = 0.0001 * np.exp(0.2 * (- epoch))
    print('lr', lr)
    return lr

callback = LearningRateScheduler(scheduler)


model.fit_generator(
    train_datagen,
    steps_per_epoch=1500,
    validation_data=val_datagen,
#     validation_steps=1921,
    validation_steps=100,
    epochs=40,
    callbacks=[callback],
)

model.save("connected_model_v2.h5")