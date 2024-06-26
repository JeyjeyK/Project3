import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions


import numpy as np
import cv2

# 사전 학습된 MobileNetV2 모델 로드
model = tf.keras.models.load_model('food/model/image_search_model.h5')

# 음식 리스트 사전 불러오기
with open('food/model/food_classes.json', 'r', encoding='utf-8') as json_file:
    food_classes = json.load(json_file)
class_label = {v: k for k, v in food_classes.items()}

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

def decode_predictions(preds):
    top_indices = preds[0].argsort()[-3:][::-1]
    result = [(class_label[i], preds[0][i]) for i in top_indices]
    return result