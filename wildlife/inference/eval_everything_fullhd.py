# prepare predictions of all models of 9 and 10 seasons to train boosting models

from pathlib import Path
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as insres_preprocess_input
import cv2
from PIL import Image
import os
import multiprocessing
DATA_PATH = Path("../../datasets/wildlife")
IMGS_PATH = '/media/ivan/data/'
# IMGS_PATH = '/media/ivan/cd486894-77c5-442b-8738-f82e04c36202/'
from tensorflow.keras.models import load_model as tf_load_model

model_480_360 = tf_load_model('insres_480_360.h5')
model_480_360_v2 = tf_load_model('insres_480_360_v2.h5')
model_512_384_v2 = tf_load_model('insres_512_384_v2.h5')
model_back = tf_load_model('background_insres_all_lr.h5')
model_mean_back_2models = tf_load_model('connected_model.h5')

for test_season in [10]:
    start = 331700
    if test_season==10:
        start = 209700
    test_season = "SER_S" + str(test_season)

    import pandas as pd

    train_metadata = pd.read_csv(DATA_PATH / "train_metadata.csv")
    train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", index_col="seq_id")

    label_columns = train_labels.columns
    train_metadata['season'] = train_metadata.seq_id.map(lambda x: x.split('#')[0])
    train_metadata = train_metadata.set_index('seq_id')

    train_metadata = train_metadata[train_metadata.season == test_season]
    train_labels = train_labels[train_labels.index.isin(train_metadata.index)]


    train_gen_df = train_labels.join(train_metadata, how='right')

    my_file = open(f"KD/everything_fullhd_{test_season}_add4.csv", "w")

    my_file.write('seq_id_buf' + "," +
                    ",".join(['pred_back_as480_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_back_as512_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_mean_back_2models_as480_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_mean_back_2models_as512_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_as480_img1_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_as480_img2_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_as480_img3_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_as512_img1_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_as512_img2_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_as512_img3_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_v2_as480_img1_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_v2_as480_img2_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_v2_as480_img3_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_v2_as512_img1_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_v2_as512_img2_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_480_360_v2_as512_img3_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_512_384_v2_as512_img1_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_512_384_v2_as512_img2_' + c for c in label_columns]) + ',' +
                    ",".join(['pred_model_512_384_v2_as512_img3_' + c for c in label_columns]) + ',' +
                  'count_imgs' + '\n')

    seq_id_buf = []
    first_img_buf_480 = []
    second_img_buf_480 = []
    third_img_buf_480 = []
    first_img_buf_512 = []
    second_img_buf_512 = []
    third_img_buf_512 = []

    imgs_back_buf_512 = []
    imgs_mean_buf_512 = []
    imgs_back_buf_480 = []
    imgs_mean_buf_480 = []

    count_imgs = []

    from copy import deepcopy
    counter = 0


    def resize_cv2(img, size):
        return cv2.resize(img, size)


    def read(path):
        if not os.path.exists(path):
            return None
        try:
            img = Image.open(path).convert("RGB")
            s1 = img.size[0] // 4
            s2 = img.size[1] // 4
            if s1 < 480 or s2 < 360:
                img = img.resize((512, 384), Image.ANTIALIAS)
            else:
                img = img.resize((s1, s2), Image.ANTIALIAS)
            return np.asarray(img)
        except:
            return None

    group = train_gen_df.groupby('seq_id').apply(lambda df: (str(df.index[0]), df['file_name'].values, df))[start:]
    least_n = len(group)
    pool = multiprocessing.Pool(3)

    for a in group:
        least_n-=1
        if least_n % 10000 == 0:
            print(least_n)
        seq_id = a[0]
        try:

            imgs = pool.map(read, [str(IMGS_PATH + "/" + p) for p in a[1][:3]])
            imgs = np.array([i for i in imgs if i is not None])
            if len(imgs) == 0:
                continue

            # remove exif
            if imgs[0].shape[0] > 450 and np.mean(imgs[0][462:, 130:320]) > 250:
                imgs[:, 458:] = 0
            if imgs[0].shape[0] > 450 and np.median(imgs[0][462:, :100]) > 200 and np.median(imgs[0][462:, :100]) < 221:
                imgs[:, 458:, :120] = 0
                imgs[:, 458:, 460:] = 0
            elif imgs[0].shape[0] > 450 and np.median(imgs[0][462:, :60]) > 200 and np.median(imgs[0][462:, :60]) < 221:
                imgs[:, 458:, :70] = 0
                imgs[:, 458:, 460:] = 0
            if np.mean(imgs[0][360:, 130:320]) > 250:
                imgs[:, 355:] = 0
            if np.median(imgs[0][365:, :100]) > 200 and np.median(imgs[0][365:, :100]) < 221:
                imgs[:, 363:, :110] = 0
                imgs[:, 363:, 330:] = 0
            elif np.median(imgs[0][365:, :60]) > 200 and np.median(imgs[0][365:, :60]) < 221:
                imgs[:, 363:, :70] = 0
                imgs[:, 363:, 330:] = 0

            # cv2.imshow('', imgs[0])
            # cv2.waitKey(0)



            first_img_buf_512.append(insres_preprocess_input(resize_cv2(imgs[0], (512,384))))
            first_img_buf_480.append(insres_preprocess_input(resize_cv2(imgs[0], (480,360))))
            if len(imgs) > 1:
                second_img_buf_512.append(insres_preprocess_input(resize_cv2(imgs[1], (512,384))))
                second_img_buf_480.append(insres_preprocess_input(resize_cv2(imgs[1], (480,360))))
            else:
                second_img_buf_512.append(np.zeros(first_img_buf_512[0].shape))
                second_img_buf_480.append(np.zeros(first_img_buf_480[0].shape))
            if len(imgs) > 2:
                third_img_buf_512.append(insres_preprocess_input(resize_cv2(imgs[2], (512,384))))
                third_img_buf_480.append(insres_preprocess_input(resize_cv2(imgs[2], (480,360))))
            else:
                third_img_buf_512.append(np.zeros(first_img_buf_512[0].shape))
                third_img_buf_480.append(np.zeros(first_img_buf_480[0].shape))


            imgs_mean = np.mean(np.asarray(imgs), axis=0)
            imgs_back = np.sum([np.abs(img - imgs_mean) for img in imgs], axis=0)
            imgs_back = np.clip(imgs_back, 0, 255)

            imgs_mean_buf_512.append(insres_preprocess_input(resize_cv2(deepcopy(imgs_mean), (512, 384))))
            imgs_mean_buf_480.append(insres_preprocess_input(resize_cv2(deepcopy(imgs_mean), (480, 360))))
            imgs_back_buf_512.append(resize_cv2(deepcopy(imgs_back), (512, 384)) / 255)
            imgs_back_buf_480.append(resize_cv2(deepcopy(imgs_back), (480, 360)) / 255)

            count_imgs.append(len(imgs))
            seq_id_buf.append(seq_id)
        except:
            print(seq_id)

        if len(seq_id_buf) == 16 or least_n == 0:
            counter += len(seq_id_buf)
            # print(counter)

            pred_back_as480 = model_back.predict(np.array(imgs_back_buf_480))  # +
            pred_back_as512 = model_back.predict(np.array(imgs_back_buf_512))  # ?
            pred_mean_back_2models_as480 = model_mean_back_2models.predict(
                [np.array(imgs_back_buf_480), np.array(imgs_mean_buf_480)])  # +
            pred_mean_back_2models_as512 = model_mean_back_2models.predict(
                [np.array(imgs_back_buf_512), np.array(imgs_mean_buf_512)])  # ?
            pred_model_480_360_as480_img1 = model_480_360.predict(np.array(first_img_buf_480))  # +
            pred_model_480_360_as480_img2 = model_480_360.predict(np.array(second_img_buf_480))  # +
            pred_model_480_360_as480_img3 = model_480_360.predict(np.array(third_img_buf_480))  # +
            pred_model_480_360_as512_img1 = model_480_360.predict(np.array(first_img_buf_512))  # ?
            pred_model_480_360_as512_img2 = model_480_360.predict(np.array(second_img_buf_512))  # ?
            pred_model_480_360_as512_img3 = model_480_360.predict(np.array(third_img_buf_512))  # ?
            pred_model_480_360_v2_as480_img1 = model_480_360_v2.predict(np.array(first_img_buf_480))  # +
            pred_model_480_360_v2_as480_img2 = model_480_360_v2.predict(np.array(second_img_buf_480))  # +
            pred_model_480_360_v2_as480_img3 = model_480_360_v2.predict(np.array(third_img_buf_480))  # +
            pred_model_480_360_v2_as512_img1 = model_480_360_v2.predict(np.array(first_img_buf_512))  # +
            pred_model_480_360_v2_as512_img2 = model_480_360_v2.predict(np.array(second_img_buf_512))  # +
            pred_model_480_360_v2_as512_img3 = model_480_360_v2.predict(np.array(third_img_buf_512))  # +
            pred_model_512_384_v2_as512_img1 = model_512_384_v2.predict(np.array(first_img_buf_512))  # ?
            pred_model_512_384_v2_as512_img2 = model_512_384_v2.predict(np.array(second_img_buf_512))  # ?
            pred_model_512_384_v2_as512_img3 = model_512_384_v2.predict(np.array(third_img_buf_512))  # ?


            for i in range(len(seq_id_buf)):
                my_file.write(seq_id_buf[i] + "," +
                              ",".join([f'{x:0.12f}' for x in pred_back_as480[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_back_as512[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_mean_back_2models_as480[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_mean_back_2models_as512[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_as480_img1[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_as480_img2[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_as480_img3[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_as512_img1[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_as512_img2[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_as512_img3[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_v2_as480_img1[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_v2_as480_img2[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_v2_as480_img3[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_v2_as512_img1[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_v2_as512_img2[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_480_360_v2_as512_img3[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_512_384_v2_as512_img1[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_512_384_v2_as512_img2[i]]) + "," +
                              ",".join([f'{x:0.12f}' for x in pred_model_512_384_v2_as512_img3[i]]) + "," +
                              str(count_imgs[i]) + '\n')

            seq_id_buf = []

            first_img_buf_480 = []
            second_img_buf_480 = []
            third_img_buf_480 = []

            first_img_buf_512 = []
            second_img_buf_512 = []
            third_img_buf_512 = []

            imgs_back_buf_512 = []
            imgs_mean_buf_512 = []

            imgs_back_buf_480 = []
            imgs_mean_buf_480 = []

            count_imgs = []

    my_file.close()

















