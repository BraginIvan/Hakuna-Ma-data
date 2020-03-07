from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import log_loss
from tensorflow.nn import swish
import tensorflow
from PIL import Image
# from tensorflow.keras.models import load_model as tf_load_model
# from keras.applications.nasnet import preprocess_input as nasnet_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_preprocess_input

def inject_tfkeras_modules(func):
    import tensorflow.keras as tfkeras
    def wrapper(*args, **kwargs):
        kwargs['backend'] = tfkeras.backend
        kwargs['layers'] = tfkeras.layers
        kwargs['models'] = tfkeras.models
        kwargs['utils'] = tfkeras.utils
        return func(*args, **kwargs)
    return wrapper

def get_dropout(**kwargs):
    backend, layers, models, keras_utils = tensorflow.keras.backend,tensorflow.keras.layers,tensorflow.keras.models,tensorflow.keras.utils

    class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout

DATA_PATH = Path("../../datasets/wildlife")

test_season = 10
test_season = "SER_S" + str(test_season)

import pandas as pd
train_metadata = pd.read_csv(DATA_PATH / "train_metadata.csv")
train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", index_col="seq_id")

train_metadata['season'] = train_metadata.seq_id.map(lambda x: x.split('#')[0])
train_metadata = train_metadata.sort_values('file_name').groupby('seq_id').first()

train_metadata = train_metadata[train_metadata.season == test_season]
train_labels = train_labels[train_labels.index.isin(train_metadata.index)]

DATAET_PATH = Path("/media/ivan/data/")

train_metadata['file_name'] = train_metadata.apply(
    lambda x: (DATA_PATH / x.file_name), axis=1
)

gen_df = train_labels.join(train_metadata.file_name.apply(lambda path: str(path)))
label_columns = train_labels.columns.tolist()

from tensorflow.keras.models import load_model as k_load_model
from keras_applications import imagenet_utils

@inject_tfkeras_modules
def preprocess_input(*args, **kwargs):
    return imagenet_utils.preprocess_input(*args, **kwargs)

def eff_preprocess_input(x, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in ['backend', 'layers', 'models', 'utils']}

    return preprocess_input(x, mode='torch', **kwargs)


# insres_512_384 = {
#         'filename': 'insres_512_384.h5',
#         'load_model': k_load_model,
#         'preprocess_input': inception_resnet_v2_preprocess_input,
#         'img_size': (384, 512)
#     }

# /home/ivan/projects/Hakuna-Ma-data/insres_360_v2.h5


insres_360 = {
        'filename': 'insres_360_v2.h5',
        'load_model': load_model,
        'preprocess_input': inception_resnet_v2_preprocess_input,
        'img_size': (480, 640)
    }


insres_480_360_v2_HUGE = {
        'filename': 'insres_480_360_v2.h5',
        'load_model': k_load_model,
        'preprocess_input': inception_resnet_v2_preprocess_input,
        'img_size': (480, 640)
    }

insres_480_360_v2_BIG = {
        'filename': 'insres_480_360_v2.h5',
        'load_model': k_load_model,
        'preprocess_input': inception_resnet_v2_preprocess_input,
        'img_size': (384, 512)
    }

insres_480_360_v2 = {
        'filename': 'insres_480_360_v2.h5',
        'load_model': k_load_model,
        'preprocess_input': inception_resnet_v2_preprocess_input,
        'img_size': (360, 480)
    }

insres_480_360 = {
        'filename': 'insres_480_360.h5',
        'load_model': k_load_model,
        'preprocess_input': inception_resnet_v2_preprocess_input,
        'img_size': (360, 480)
    }

insres = {
        'filename': 'insres.h5',
        'load_model': k_load_model,
        'preprocess_input': inception_resnet_v2_preprocess_input,
        'img_size': (299, 299)
    }

Xception_sigmoid_lr2 = {
        'filename': 'Xception_sigmoid_lr2.h5',
        'load_model': k_load_model,
        'preprocess_input': xception_preprocess_input,
        'img_size': (299, 299)
    }

Xception_sigmoid_lr = {
        'filename': 'Xception_sigmoid_lr.h5',
        'load_model': k_load_model,
        'preprocess_input': xception_preprocess_input,
        'img_size': (299, 299)
    }

Xception_sigmoid = {
        'filename': 'Xception_sigmoid.h5',
        'load_model': k_load_model,
        'preprocess_input': xception_preprocess_input,
        'img_size': (299, 299)
    }



B3_340_260 = {
        'filename': 'B3_340_260.h5',
        'load_model': k_load_model,
        'preprocess_input': eff_preprocess_input,
        'img_size': (260, 340)
    }

B3_300_2 = {
        'filename': 'B3_300_2.h5',
        'load_model': k_load_model,
        'preprocess_input': eff_preprocess_input,
        'img_size': (260,260)
    }


B3_final2 = {
        'filename': 'B3_final2.h5',
        'load_model': k_load_model,
        'preprocess_input': eff_preprocess_input,
        'img_size': (260,260)
    }

B3_final = {
        'filename': 'B3_final.h5',
        'load_model': k_load_model,
        'preprocess_input': eff_preprocess_input,
        'img_size': (260,260)
    }

B3_augm = {
        'filename': 'B3_augm.h5', #B3_no_dropout.h5
        'load_model': k_load_model,
        'preprocess_input': eff_preprocess_input,
        'img_size': (260,260)
    }

my_awesome_model = {
        'filename': 'my_awesome_model.h5', #B3_no_dropout.h5
        'load_model': k_load_model,
        'preprocess_input': eff_preprocess_input,
        'img_size': (260,260)
    }

B3_cee_all_data_sigmoid = {
        'filename': 'B3_cee_all_data_sigmoid.h5',
        'load_model': k_load_model,
        'preprocess_input': eff_preprocess_input,
        'img_size': (260,260)
    }


# Xception = {
#         'filename': 'Xception_cee.h5',
#         'load_model': tf_load_model,
#         'preprocess_input': xception_preprocess_input,
#         'img_size': 299
#     }

import cv2
for d in [insres_480_360_v2_BIG, insres_480_360_v2_HUGE]:

    datagen = ImageDataGenerator(
        preprocessing_function=d['preprocess_input'],
    )
    target_size = d['img_size']



    gen = datagen.flow_from_dataframe(
        dataframe=gen_df,
        x_col="file_name",
        y_col=label_columns,
        class_mode="other",
        target_size=target_size,
        batch_size=32,
        shuffle=False
    )
    filenames = gen.filenames
    labels = gen.labels
    nb_samples = len(filenames)


    model = d['load_model'](d['filename'], custom_objects={"swish": swish, 'FixedDropout': get_dropout()})



    print('predict')

    # predict = model.predict_generator(gen, steps=np.ceil(nb_samples/32), verbose=1)
    predict = model.predict_generator(gen, steps=300, verbose=1)

    print('done')


    score = 0

    for i in range(54):
        if sum(labels[:len(predict), i].astype(np.float64)) != 0:
            x = log_loss(labels[:len(predict), i].astype(np.float64), predict[:,i].reshape((-1)).astype(np.float64))
            print(label_columns[i] + " " + str(x))
            score += x
        else:
            print(label_columns[i], "not found")
    print(d['filename'], score / 54)


