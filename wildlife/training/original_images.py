import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Dropout
import sys


from tensorflow.keras.applications import inception_resnet_v2
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

version = sys.argv[2]


class Pipeline:
    def __init__(self,resolution,batch_size,epoches,start_lr):
        self.resolution = resolution
        self.batch_size = batch_size
        self.epoches = epoches
        self.start_lr = start_lr


if version == '0':
    pipelines = [
        Pipeline(resolution=(224, 224),
                 batch_size=48,
                 epoches=20,
                 start_lr=0.0004
                 ),
        Pipeline(resolution=(299, 299),
                 batch_size=32,
                 epoches=20,
                 start_lr=0.0001
                 ),
        Pipeline(resolution=(360, 480),
                 batch_size=16,
                 epoches=15,
                 start_lr=0.00003
                 ),
        Pipeline(resolution=(384, 512),
                 batch_size=16,
                 epoches=15,
                 start_lr=0.00001
                 )]
    train_steps = 10
    val_steps = 10
    evaluation_steps = 100
elif version == '1':
    pipelines = [
        Pipeline(resolution=(240, 320),
                 batch_size=32,
                 epoches=20,
                 start_lr=0.0003
                 ),
        Pipeline(resolution=(360, 480),
                 batch_size=16,
                 epoches=15,
                 start_lr=0.00003
                 ),
        Pipeline(resolution=(384, 512),
                 batch_size=16,
                 epoches=15,
                 start_lr=0.00001
                 )]
    train_steps = 10000
    val_steps = 100
    evaluation_steps = 20000
elif version == '2':
    pipelines = [
        Pipeline(resolution=(224, 224),
                 batch_size=48,
                 epoches=25,
                 start_lr=0.0004
                 ),
        Pipeline(resolution=(299, 299),
                 batch_size=32,
                 epoches=20,
                 start_lr=0.0001
                 ),
        Pipeline(resolution=(360, 480),
                 batch_size=16,
                 epoches=15,
                 start_lr=0.00003
                 ),
        Pipeline(resolution=(384, 512),
                 batch_size=16,
                 epoches=15,
                 start_lr=0.00001
                 )]
    train_steps = 5000
    val_steps = 100
    evaluation_steps = 20000
else:
    resolutions = []
    epoch_train_steps = 0
    epoch_val_steps = 0
    epoches_on_resolution = 0
    evaluation_steps = 0
    raise AttributeError('version expected 0 or 1 or 2')


DATA_PATH = Path(sys.argv[1])

train_metadata = pd.read_csv(DATA_PATH / "train_metadata.csv")
train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", index_col="seq_id")

train_metadata = train_metadata.sort_values('file_name').groupby('seq_id').first()
train_metadata['season'] = train_metadata.index.map(lambda x: x.split('#')[0])

train_metadata['file_name'] = train_metadata.apply(
    lambda x: (DATA_PATH / x.file_name), axis=1
)

train_seasons = ['SER_S1', 'SER_S2', "SER_S3", 'SER_S4', 'SER_S5', 'SER_S6', 'SER_S7', 'SER_S8']
val_seasons = ["SER_S9"] # season 10 will be used on the next stage

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

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_gens(target_size, batch_size):
    train_datagen = datagen_flip.flow_from_dataframe(
        dataframe=train_gen_df,
        x_col="file_name",
        y_col=label_columns,
        class_mode="other",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_datagen = datagen.flow_from_dataframe(
        dataframe=val_gen_df,
        x_col="file_name",
        y_col=label_columns,
        class_mode="other",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True
    )
    return train_datagen, val_datagen

def get_scheduler(start_lr):
    def scheduler(epoch):
        lr = start_lr * np.exp(0.15 * (- epoch))
        print("lr =", lr)
        return lr
    return scheduler

transfer = inception_resnet_v2.InceptionResNetV2(include_top=False)
max_pooling = GlobalMaxPooling2D(name="max_pooling")(transfer.output)
drop_out = Dropout(0.05, name="dropout1")(max_pooling)
outputs = Dense(len(label_columns), activation="sigmoid")(drop_out)
model = Model(inputs=transfer.input, outputs=outputs)

for pipeline in pipelines:
    callback = LearningRateScheduler(get_scheduler(pipeline.start_lr))

    train_datagen, val_datagen = get_gens(pipeline.resolution, pipeline.batch_size)
    for layersN in [1, 100, 200]:
        print("resolution {} last {} layers".format(pipeline.resolution, layersN))
        for layer in model.layers:
            layer.trainable = False
        for layer in model.layers[-layersN:]:
            layer.trainable = True

        opt = Adam(0.00002)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[categorical_accuracy])
        model.fit_generator(
            train_datagen,
            steps_per_epoch=train_steps,
            validation_data=val_datagen,
            validation_steps=val_steps,
            epochs=1)

    for layer in model.layers:
        layer.trainable = True

    print("resolution {} all layers".format(pipeline.resolution))

    opt = Adam()
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[categorical_accuracy])
    model.fit_generator(
        train_datagen,
        steps_per_epoch=train_steps,
        validation_data=val_datagen,
        validation_steps=val_steps,
        callbacks=[callback],
        epochs=pipeline.epoches
    )
    model_name = "insres_{}_v{}".format(pipeline.resolution[0], version)   + ".h5"
    model.save(model_name)
    print(model_name, 'evaluation')
    model.evaluate(val_datagen, steps=evaluation_steps)


# Found 367470 validated image filenames.
# resolution (224, 224) last 1 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 24s 2s/step - loss: 1.1200 - categorical_accuracy: 0.0063 - val_loss: 1.1434 - val_categorical_accuracy: 0.0146
# resolution (224, 224) last 100 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 22s 2s/step - loss: 0.9270 - categorical_accuracy: 0.0229 - val_loss: 0.8257 - val_categorical_accuracy: 0.0417
# resolution (224, 224) last 200 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 23s 2s/step - loss: 0.7203 - categorical_accuracy: 0.0437 - val_loss: 0.6136 - val_categorical_accuracy: 0.0812
# resolution (224, 224) all layers
# Train for 10 steps, validate for 10 steps
# lr = 0.0004
# Epoch 1/30
# 10/10 [==============================] - 35s 3s/step - loss: 0.1561 - categorical_accuracy: 0.5646 - val_loss: 0.0359 - val_categorical_accuracy: 0.7646
# lr = 0.00034428319057002316
# Epoch 2/30
# 10/10 [==============================] - 8s 757ms/step - loss: 0.0361 - categorical_accuracy: 0.7521 - val_loss: 0.0252 - val_categorical_accuracy: 0.8646
# lr = 0.00029632728827268717
# Epoch 3/30
# 10/10 [==============================] - 8s 763ms/step - loss: 0.0299 - categorical_accuracy: 0.8042 - val_loss: 0.0202 - val_categorical_accuracy: 0.8562
# lr = 0.00025505126064870933
# Epoch 4/30
# 10/10 [==============================] - 8s 770ms/step - loss: 0.0262 - categorical_accuracy: 0.8000 - val_loss: 0.0189 - val_categorical_accuracy: 0.8833
# lr = 0.00021952465443761057
# Epoch 5/30
# 10/10 [==============================] - 7s 745ms/step - loss: 0.0222 - categorical_accuracy: 0.8396 - val_loss: 0.0170 - val_categorical_accuracy: 0.8813
# lr = 0.0001889466210964059
# Epoch 6/30
# 10/10 [==============================] - 8s 752ms/step - loss: 0.0214 - categorical_accuracy: 0.8458 - val_loss: 0.0159 - val_categorical_accuracy: 0.8917
# lr = 0.0001626278638962397
# Epoch 7/30
# 10/10 [==============================] - 7s 749ms/step - loss: 0.0259 - categorical_accuracy: 0.8021 - val_loss: 0.0162 - val_categorical_accuracy: 0.8813
# lr = 0.00013997509964446214
# Epoch 8/30
# 10/10 [==============================] - 8s 764ms/step - loss: 0.0212 - categorical_accuracy: 0.8396 - val_loss: 0.0160 - val_categorical_accuracy: 0.8875
# lr = 0.00012047768476488086
# Epoch 9/30
# 10/10 [==============================] - 7s 742ms/step - loss: 0.0165 - categorical_accuracy: 0.8813 - val_loss: 0.0158 - val_categorical_accuracy: 0.8917
# lr = 0.00010369610425835663
# Epoch 10/30
# 10/10 [==============================] - 8s 766ms/step - loss: 0.0192 - categorical_accuracy: 0.8583 - val_loss: 0.0149 - val_categorical_accuracy: 0.9000
# lr = 8.925206405937193e-05
# Epoch 11/30
# 10/10 [==============================] - 7s 740ms/step - loss: 0.0190 - categorical_accuracy: 0.8729 - val_loss: 0.0154 - val_categorical_accuracy: 0.8896
# lr = 7.681996344830166e-05
# Epoch 12/30
# 10/10 [==============================] - 7s 746ms/step - loss: 0.0231 - categorical_accuracy: 0.8271 - val_loss: 0.0146 - val_categorical_accuracy: 0.8958
# lr = 6.611955528863462e-05
# Epoch 13/30
# 10/10 [==============================] - 8s 770ms/step - loss: 0.0206 - categorical_accuracy: 0.8250 - val_loss: 0.0147 - val_categorical_accuracy: 0.8875
# lr = 5.690962863460544e-05
# Epoch 14/30
# 10/10 [==============================] - 7s 745ms/step - loss: 0.0218 - categorical_accuracy: 0.8333 - val_loss: 0.0133 - val_categorical_accuracy: 0.9125
# lr = 4.898257130119277e-05
# Epoch 15/30
# 10/10 [==============================] - 8s 767ms/step - loss: 0.0186 - categorical_accuracy: 0.8625 - val_loss: 0.0133 - val_categorical_accuracy: 0.9104
# lr = 4.2159689824745734e-05
# Epoch 16/30
# 10/10 [==============================] - 8s 760ms/step - loss: 0.0164 - categorical_accuracy: 0.8833 - val_loss: 0.0129 - val_categorical_accuracy: 0.9104
# lr = 3.628718131576501e-05
# Epoch 17/30
# 10/10 [==============================] - 7s 745ms/step - loss: 0.0138 - categorical_accuracy: 0.8917 - val_loss: 0.0126 - val_categorical_accuracy: 0.9104
# lr = 3.123266640046127e-05
# Epoch 18/30
# 10/10 [==============================] - 8s 768ms/step - loss: 0.0184 - categorical_accuracy: 0.8521 - val_loss: 0.0125 - val_categorical_accuracy: 0.9167
# lr = 2.6882205095899913e-05
# Epoch 19/30
# 10/10 [==============================] - 8s 791ms/step - loss: 0.0162 - categorical_accuracy: 0.8708 - val_loss: 0.0126 - val_categorical_accuracy: 0.9125
# lr = 2.3137728349935383e-05
# Epoch 20/30
# 10/10 [==============================] - 8s 764ms/step - loss: 0.0189 - categorical_accuracy: 0.8417 - val_loss: 0.0128 - val_categorical_accuracy: 0.9167
# lr = 1.991482734714558e-05
# Epoch 21/30
# 10/10 [==============================] - 8s 756ms/step - loss: 0.0200 - categorical_accuracy: 0.8562 - val_loss: 0.0128 - val_categorical_accuracy: 0.9167
# lr = 1.7140850746816075e-05
# Epoch 22/30
# 10/10 [==============================] - 8s 768ms/step - loss: 0.0179 - categorical_accuracy: 0.8604 - val_loss: 0.0125 - val_categorical_accuracy: 0.9167
# lr = 1.4753266960496006e-05
# Epoch 23/30
# 10/10 [==============================] - 8s 752ms/step - loss: 0.0185 - categorical_accuracy: 0.8562 - val_loss: 0.0122 - val_categorical_accuracy: 0.9167
# lr = 1.2698254551227181e-05
# Epoch 24/30
# 10/10 [==============================] - 8s 762ms/step - loss: 0.0178 - categorical_accuracy: 0.8750 - val_loss: 0.0121 - val_categorical_accuracy: 0.9187
# lr = 1.0929488978917029e-05
# Epoch 25/30
# 10/10 [==============================] - 8s 766ms/step - loss: 0.0146 - categorical_accuracy: 0.8979 - val_loss: 0.0121 - val_categorical_accuracy: 0.9167
# lr = 9.407098342403643e-06
# Epoch 26/30
# 10/10 [==============================] - 7s 708ms/step - loss: 0.0203 - categorical_accuracy: 0.8167 - val_loss: 0.0120 - val_categorical_accuracy: 0.9208
# lr = 8.096764578321757e-06
# Epoch 27/30
# 10/10 [==============================] - 7s 735ms/step - loss: 0.0156 - categorical_accuracy: 0.8813 - val_loss: 0.0120 - val_categorical_accuracy: 0.9208
# lr = 6.9689498557974066e-06
# Epoch 28/30
# 10/10 [==============================] - 8s 767ms/step - loss: 0.0168 - categorical_accuracy: 0.8771 - val_loss: 0.0121 - val_categorical_accuracy: 0.9208
# lr = 5.998230728191081e-06
# Epoch 29/30
# 10/10 [==============================] - 7s 726ms/step - loss: 0.0152 - categorical_accuracy: 0.8813 - val_loss: 0.0122 - val_categorical_accuracy: 0.9208
# lr = 5.162725032191949e-06
# Epoch 30/30
# 10/10 [==============================] - 7s 747ms/step - loss: 0.0197 - categorical_accuracy: 0.8375 - val_loss: 0.0121 - val_categorical_accuracy: 0.9208
# insres_224_v0.h5 evaluation
# 100/100 [==============================] - 37s 373ms/step - loss: 0.0147 - categorical_accuracy: 0.8998
# Found 1769267 validated image filenames.
# Found 367470 validated image filenames.
# resolution (299, 299) last 1 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 21s 2s/step - loss: 0.0189 - categorical_accuracy: 0.8531 - val_loss: 0.0193 - val_categorical_accuracy: 0.8938
# resolution (299, 299) last 100 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 21s 2s/step - loss: 0.0176 - categorical_accuracy: 0.8969 - val_loss: 0.0169 - val_categorical_accuracy: 0.8938
# resolution (299, 299) last 200 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 22s 2s/step - loss: 0.0214 - categorical_accuracy: 0.8281 - val_loss: 0.0160 - val_categorical_accuracy: 0.8969
# resolution (299, 299) all layers
# Train for 10 steps, validate for 10 steps
# lr = 0.0001
# Epoch 1/20
# 10/10 [==============================] - 36s 4s/step - loss: 0.0217 - categorical_accuracy: 0.8469 - val_loss: 0.0156 - val_categorical_accuracy: 0.8906
# lr = 8.607079764250579e-05
# Epoch 2/20
# 10/10 [==============================] - 8s 751ms/step - loss: 0.0190 - categorical_accuracy: 0.8656 - val_loss: 0.0201 - val_categorical_accuracy: 0.8781
# lr = 7.408182206817179e-05
# Epoch 3/20
# 10/10 [==============================] - 8s 767ms/step - loss: 0.0176 - categorical_accuracy: 0.8781 - val_loss: 0.0167 - val_categorical_accuracy: 0.8875
# lr = 6.376281516217733e-05
# Epoch 4/20
# 10/10 [==============================] - 8s 786ms/step - loss: 0.0170 - categorical_accuracy: 0.8687 - val_loss: 0.0157 - val_categorical_accuracy: 0.8906
# lr = 5.488116360940264e-05
# Epoch 5/20
# 10/10 [==============================] - 8s 790ms/step - loss: 0.0154 - categorical_accuracy: 0.8906 - val_loss: 0.0160 - val_categorical_accuracy: 0.8938
# lr = 4.723665527410147e-05
# Epoch 6/20
# 10/10 [==============================] - 8s 780ms/step - loss: 0.0143 - categorical_accuracy: 0.8938 - val_loss: 0.0145 - val_categorical_accuracy: 0.9000
# lr = 4.065696597405992e-05
# Epoch 7/20
# 10/10 [==============================] - 8s 762ms/step - loss: 0.0152 - categorical_accuracy: 0.8875 - val_loss: 0.0152 - val_categorical_accuracy: 0.8875
# lr = 3.4993774911115536e-05
# Epoch 8/20
# 10/10 [==============================] - 8s 790ms/step - loss: 0.0172 - categorical_accuracy: 0.8656 - val_loss: 0.0150 - val_categorical_accuracy: 0.8875
# lr = 3.0119421191220214e-05
# Epoch 9/20
# 10/10 [==============================] - 8s 779ms/step - loss: 0.0152 - categorical_accuracy: 0.8750 - val_loss: 0.0136 - val_categorical_accuracy: 0.8969
# lr = 2.5924026064589158e-05
# Epoch 10/20
# 10/10 [==============================] - 8s 788ms/step - loss: 0.0187 - categorical_accuracy: 0.8719 - val_loss: 0.0127 - val_categorical_accuracy: 0.9031
# lr = 2.2313016014842984e-05
# Epoch 11/20
# 10/10 [==============================] - 8s 767ms/step - loss: 0.0127 - categorical_accuracy: 0.9031 - val_loss: 0.0131 - val_categorical_accuracy: 0.9000
# lr = 1.9204990862075415e-05
# Epoch 12/20
# 10/10 [==============================] - 8s 777ms/step - loss: 0.0180 - categorical_accuracy: 0.8562 - val_loss: 0.0130 - val_categorical_accuracy: 0.9062
# lr = 1.6529888822158655e-05
# Epoch 13/20
# 10/10 [==============================] - 8s 774ms/step - loss: 0.0173 - categorical_accuracy: 0.8594 - val_loss: 0.0130 - val_categorical_accuracy: 0.9094
# lr = 1.422740715865136e-05
# Epoch 14/20
# 10/10 [==============================] - 8s 771ms/step - loss: 0.0170 - categorical_accuracy: 0.8656 - val_loss: 0.0137 - val_categorical_accuracy: 0.9094
# lr = 1.2245642825298192e-05
# Epoch 15/20
# 10/10 [==============================] - 8s 773ms/step - loss: 0.0192 - categorical_accuracy: 0.8531 - val_loss: 0.0141 - val_categorical_accuracy: 0.9125
# lr = 1.0539922456186433e-05
# Epoch 16/20
# 10/10 [==============================] - 8s 778ms/step - loss: 0.0151 - categorical_accuracy: 0.8781 - val_loss: 0.0143 - val_categorical_accuracy: 0.9062
# lr = 9.071795328941252e-06
# Epoch 17/20
# 10/10 [==============================] - 8s 789ms/step - loss: 0.0148 - categorical_accuracy: 0.8969 - val_loss: 0.0141 - val_categorical_accuracy: 0.9031
# lr = 7.808166600115318e-06
# Epoch 18/20
# 10/10 [==============================] - 8s 761ms/step - loss: 0.0146 - categorical_accuracy: 0.8750 - val_loss: 0.0146 - val_categorical_accuracy: 0.9031
# lr = 6.720551273974978e-06
# Epoch 19/20
# 10/10 [==============================] - 8s 779ms/step - loss: 0.0153 - categorical_accuracy: 0.8969 - val_loss: 0.0145 - val_categorical_accuracy: 0.9031
# lr = 5.784432087483846e-06
# Epoch 20/20
# 10/10 [==============================] - 8s 777ms/step - loss: 0.0145 - categorical_accuracy: 0.8719 - val_loss: 0.0146 - val_categorical_accuracy: 0.9031
# insres_299_v0.h5 evaluation
# 100/100 [==============================] - 31s 305ms/step - loss: 0.0147 - categorical_accuracy: 0.9141
# Found 1769267 validated image filenames.
# Found 367470 validated image filenames.
# resolution (360, 480) last 1 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 18s 2s/step - loss: 0.0170 - categorical_accuracy: 0.8750 - val_loss: 0.0082 - val_categorical_accuracy: 0.9563
# resolution (360, 480) last 100 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 21s 2s/step - loss: 0.0118 - categorical_accuracy: 0.9187 - val_loss: 0.0088 - val_categorical_accuracy: 0.9500
# resolution (360, 480) last 200 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 20s 2s/step - loss: 0.0090 - categorical_accuracy: 0.9250 - val_loss: 0.0092 - val_categorical_accuracy: 0.9500
# resolution (360, 480) all layers
# Train for 10 steps, validate for 10 steps
# lr = 3e-05
# Epoch 1/10
# 10/10 [==============================] - 34s 3s/step - loss: 0.0203 - categorical_accuracy: 0.8625 - val_loss: 0.0091 - val_categorical_accuracy: 0.9500
# lr = 2.5821239292751733e-05
# Epoch 2/10
# 10/10 [==============================] - 7s 692ms/step - loss: 0.0099 - categorical_accuracy: 0.9312 - val_loss: 0.0089 - val_categorical_accuracy: 0.9438
# lr = 2.2224546620451535e-05
# Epoch 3/10
# 10/10 [==============================] - 7s 690ms/step - loss: 0.0135 - categorical_accuracy: 0.8938 - val_loss: 0.0085 - val_categorical_accuracy: 0.9500
# lr = 1.91288445486532e-05
# Epoch 4/10
# 10/10 [==============================] - 7s 692ms/step - loss: 0.0152 - categorical_accuracy: 0.8750 - val_loss: 0.0081 - val_categorical_accuracy: 0.9563
# lr = 1.6464349082820793e-05
# Epoch 5/10
# 10/10 [==============================] - 7s 693ms/step - loss: 0.0135 - categorical_accuracy: 0.8687 - val_loss: 0.0082 - val_categorical_accuracy: 0.9563
# lr = 1.417099658223044e-05
# Epoch 6/10
# 10/10 [==============================] - 7s 691ms/step - loss: 0.0150 - categorical_accuracy: 0.8875 - val_loss: 0.0081 - val_categorical_accuracy: 0.9563
# lr = 1.2197089792217975e-05
# Epoch 7/10
# 10/10 [==============================] - 7s 706ms/step - loss: 0.0128 - categorical_accuracy: 0.9125 - val_loss: 0.0084 - val_categorical_accuracy: 0.9563
# lr = 1.049813247333466e-05
# Epoch 8/10
# 10/10 [==============================] - 7s 701ms/step - loss: 0.0187 - categorical_accuracy: 0.8687 - val_loss: 0.0084 - val_categorical_accuracy: 0.9563
# lr = 9.035826357366064e-06
# Epoch 9/10
# 10/10 [==============================] - 7s 689ms/step - loss: 0.0142 - categorical_accuracy: 0.8625 - val_loss: 0.0080 - val_categorical_accuracy: 0.9563
# lr = 7.777207819376747e-06
# Epoch 10/10
# 10/10 [==============================] - 7s 687ms/step - loss: 0.0123 - categorical_accuracy: 0.9187 - val_loss: 0.0079 - val_categorical_accuracy: 0.9563
# insres_360_v0.h5 evaluation
# 100/100 [==============================] - 21s 205ms/step - loss: 0.0127 - categorical_accuracy: 0.9150
# Found 1769267 validated image filenames.
# Found 367470 validated image filenames.
# resolution (384, 512) last 1 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 18s 2s/step - loss: 0.0175 - categorical_accuracy: 0.8813 - val_loss: 0.0142 - val_categorical_accuracy: 0.9000
# resolution (384, 512) last 100 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 19s 2s/step - loss: 0.0213 - categorical_accuracy: 0.8250 - val_loss: 0.0142 - val_categorical_accuracy: 0.9000
# resolution (384, 512) last 200 layers
# Train for 10 steps, validate for 10 steps
# 10/10 [==============================] - 22s 2s/step - loss: 0.0131 - categorical_accuracy: 0.8750 - val_loss: 0.0151 - val_categorical_accuracy: 0.8938
# resolution (384, 512) all layers
# Train for 10 steps, validate for 10 steps
# lr = 1e-05
# Epoch 1/10
# 10/10 [==============================] - 35s 3s/step - loss: 0.0143 - categorical_accuracy: 0.9062 - val_loss: 0.0153 - val_categorical_accuracy: 0.8938
# lr = 8.60707976425058e-06
# Epoch 2/10
# 10/10 [==============================] - 8s 766ms/step - loss: 0.0162 - categorical_accuracy: 0.8938 - val_loss: 0.0148 - val_categorical_accuracy: 0.8938
# lr = 7.408182206817179e-06
# Epoch 3/10
# 10/10 [==============================] - 8s 765ms/step - loss: 0.0155 - categorical_accuracy: 0.8938 - val_loss: 0.0143 - val_categorical_accuracy: 0.8938
# lr = 6.376281516217734e-06
# Epoch 4/10
# 10/10 [==============================] - 8s 791ms/step - loss: 0.0194 - categorical_accuracy: 0.8687 - val_loss: 0.0139 - val_categorical_accuracy: 0.8875
# lr = 5.488116360940264e-06
# Epoch 5/10
# 10/10 [==============================] - 8s 773ms/step - loss: 0.0163 - categorical_accuracy: 0.8938 - val_loss: 0.0137 - val_categorical_accuracy: 0.9000
# lr = 4.723665527410147e-06
# Epoch 6/10
# 10/10 [==============================] - 8s 792ms/step - loss: 0.0120 - categorical_accuracy: 0.9187 - val_loss: 0.0137 - val_categorical_accuracy: 0.9000
# lr = 4.065696597405992e-06
# Epoch 7/10
# 10/10 [==============================] - 8s 773ms/step - loss: 0.0149 - categorical_accuracy: 0.8938 - val_loss: 0.0138 - val_categorical_accuracy: 0.9000
# lr = 3.4993774911115535e-06
# Epoch 8/10
# 10/10 [==============================] - 8s 779ms/step - loss: 0.0166 - categorical_accuracy: 0.8875 - val_loss: 0.0140 - val_categorical_accuracy: 0.9000
# lr = 3.0119421191220218e-06
# Epoch 9/10
# 10/10 [==============================] - 8s 778ms/step - loss: 0.0170 - categorical_accuracy: 0.8687 - val_loss: 0.0139 - val_categorical_accuracy: 0.9000
# lr = 2.592402606458916e-06
# Epoch 10/10
# 10/10 [==============================] - 8s 798ms/step - loss: 0.0168 - categorical_accuracy: 0.8750 - val_loss: 0.0137 - val_categorical_accuracy: 0.9000
# insres_384_v0.h5 evaluation
# 100/100 [==============================] - 24s 235ms/step - loss: 0.0129 - categorical_accuracy: 0.9056

