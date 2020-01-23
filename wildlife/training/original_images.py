from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Input, \
    Lambda

from tensorflow.keras.applications import inception_resnet_v2

DATA_PATH = Path("/home/ivan/projects/datasets/wildlife")

train_metadata = pd.read_csv(DATA_PATH / "train_metadata.csv")
train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", index_col="seq_id")

train_metadata = train_metadata.sort_values('file_name').groupby('seq_id').first()
train_metadata['season'] = train_metadata.index.map(lambda x: x.split('#')[0])

train_metadata['file_name'] = train_metadata.apply(
    lambda x: (DATA_PATH / x.file_name), axis=1
)

train_seasons = ['SER_S1', 'SER_S2', "SER_S3", 'SER_S4', 'SER_S5', 'SER_S6', 'SER_S7', 'SER_S8']
val_seasons = ["SER_S9"]

# split out validation first
val_x = train_metadata[train_metadata.season.isin(val_seasons)]
val_y = train_labels[train_labels.index.isin(val_x.index)]

# reduce training
train_metadata = train_metadata[train_metadata.season.isin(train_seasons)]
train_labels = train_labels[train_labels.index.isin(train_metadata.index)]

train_gen_df = train_labels.join(train_metadata.file_name.apply(lambda path: str(path)))
val_gen_df = val_y.join(val_x.file_name.apply(lambda path: str(path)))
label_columns = train_labels.columns.tolist()

datagen_flip = ImageDataGenerator(preprocessing_function=inception_resnet_v2.preprocess_input, horizontal_flip=True)
datagen = ImageDataGenerator(preprocessing_function=inception_resnet_v2.preprocess_input)

target_size = (360, 480)
ImageFile.LOAD_TRUNCATED_IMAGES = True
train_datagen = datagen_flip.flow_from_dataframe(
    dataframe=train_gen_df,
    x_col="file_name",
    y_col=label_columns,
    class_mode="other",
    target_size=target_size,
    batch_size=16,
    shuffle=True
)

val_datagen = datagen.flow_from_dataframe(
    dataframe=val_gen_df,
    x_col="file_name",
    y_col=label_columns,
    class_mode="other",
    target_size=target_size,
    batch_size=32,
    shuffle=True
)

transfer = inception_resnet_v2.InceptionResNetV2(include_top=False)
max_pooling = GlobalMaxPooling2D(name="max_pooling")(transfer.output)
outputs = Dense(len(label_columns), activation="sigmoid")(max_pooling)
model = Model(inputs=transfer.input, outputs=outputs)

for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-1:]:
    layer.trainable = True

opt = Adam()
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[categorical_accuracy])
model.fit_generator(
    train_datagen,
    steps_per_epoch=1000,
    validation_data=val_datagen,
    validation_steps=len(val_datagen),
    epochs=2)

for layer in model.layers:
    layer.trainable = True


def scheduler(epoch):
    if epoch < 5:
        return 0.0002
    else:
        return 0.0002 * np.exp(0.1 * (5 - epoch))


opt = Adam()
callback = LearningRateScheduler(scheduler)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[categorical_accuracy])
model.fit_generator(
    train_datagen,
    steps_per_epoch=5000,
    validation_data=val_datagen,
    validation_steps=len(val_datagen),
    epochs=40)

model.save("insres_%d_%d.h5" % target_size)
