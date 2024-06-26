from flask import Blueprint, render_template, request, redirect, current_app, url_for, jsonify

import unicodedata

import os
import cv2
import sys

## 그림을 위한 패키지
import base64

sys.path.append('D:\song\project\project3')

from web.food.func.model import predict_image, decode_predictions
from web.food.func.recipe import select_menu

bp = Blueprint('image', __name__, url_prefix='/image')

@bp.route('/', methods = ['GET', 'POST'])
def image_home():
    error = None
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            error = '파일을 선택하지 않았습니다. 다시 시도해 주세요.'

        else:
            return redirect(url_for('image.image_upload'))

    return render_template('image/image_search.html', error=error)

@bp.route('/image', methods = ['GET', 'POST'])
def image_upload():

    width, height = None, None
    error = None
    img_str = None
    food_name = None
    p_value = None
    step = None
    ingredient = None
    step_easy = None
    ingredient_easy = None

    file = request.files.get('image')
    if file:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is not None:
            height, width = img.shape[:2]
            preds = predict_image(filepath)
            p_value = decode_predictions(preds)
            food_name = p_value[0][0]

            food_name = unicodedata.normalize('NFC', str(food_name))
            recipe = select_menu(str(food_name))

            step = recipe['조리순서'].values[0].split("\n")
            ingredient = recipe['요리재료내용'].values[0].split('[')

            recipe2 = recipe.copy()
            recipe2 = recipe2.sort_values(by='조리순서길이', ascending=True)

            step_easy = recipe2['조리순서'].values[0].split("\n")
            ingredient_easy = recipe2['요리재료내용'].values[0].split('[')

            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode('utf-8')
        else:
            error = '이미지 파일이 아닙니다. 다시 시도해 주세요.'

    return render_template('image/image_results.html',
                           width=width, height=height, error=error,
                           img_data=img_str,
                           p_value=p_value,
                           food_name=food_name,
                           step=step,
                           ingredient=ingredient,
                           step_easy=step_easy,
                           ingredient_easy=ingredient_easy)