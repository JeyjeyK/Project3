from flask import Flask, jsonify, request, render_template
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 데이터 및 모델 로드
reci = pd.read_excel('food/model/recipe_3.xlsx')
reci = reci[['레시피제목','요리명','요리재료내용','재료','조리순서','조회수','스크랩수','요리종류별명','재료_불용어 제외']]

label_encoder = LabelEncoder()
reci['요리명_label'] = label_encoder.fit_transform(reci['요리명'])
class_to_recipe_name = dict(zip(reci['요리명_label'], reci['요리명']))

model = load_model('food/model/best_model.h5')

with open('food/model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def normalize_word(word):
    replacements = {
        "달걀": "계란",
        "오뎅": "어묵",
        "무우": "무",
        "대파": "파",
        "씨레기": "시래기",
        "쭈꾸미":"주꾸미",
        "고추가루":"고춧가루",
        "자장면":"짜장면"
    }
    return replacements.get(word.strip(), word.strip())

def prepare_text(text, tokenizer, maxlen=34):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

def show_categories(df):
    return df['요리종류별명'].unique()

def show_recipes_by_category(df, category):
    filtered_df = df[df['요리종류별명'] == category]
    return filtered_df['요리명'].unique()

def recommend_recipes(df, recipe_name):
    filtered_df = df[df['요리명'].str.contains(recipe_name)]
    sorted_df = filtered_df.sort_values(by=['조회수', '스크랩수'], ascending=[False, False])
    top_recipes = sorted_df.head(3)

    responses = []
    for i, match in enumerate(top_recipes.iterrows(), start=1):
        response = "<div class='recipe-container'>\n"
            # 아이콘을 왼쪽에 추가합니다.
        response += f"<img class='recipe-icon' src='/static/images/icon1.png' alt='Recipe Icon'>\n"
            # 응답 박스를 오른쪽에 추가합니다.
        response += f"<div class='recipe-box'>\n"
        if i == 1:
            response += f"<div style='text-align: center; margin-bottom: 10px;'><p style='color: #4a83c9; font-weight: bold; font-size: 1.5em;'  class='cooking-partner'>쿠킹파트너 : {match[1]['요리명']} 레시피를 알려드리겠습니다.</p></div>\n"

        response += f"<p class='recipe-title'><strong>{i}. {match[1]['레시피제목']}</strong></p>\n"

        # 요리 재료 내용에서 줄바꿈 구현
        ingredients = match[1]['요리재료내용'].split('\n')
        formatted_ingredients = "<br>".join(ingredients)
        response += f"<p>{formatted_ingredients}</p>\n"

        # 조리순서를 구분자인 숫자와 마침표를 기준으로 나누어 줄바꿈합니다.
        cooking_steps = match[1]['조리순서'].split('\n')
        formatted_cooking_steps = "<br>".join(cooking_steps)
        response += f"<p><span class='cooking-step'>[조리순서]</span><br>{formatted_cooking_steps}</p>\n"
        response += "</div>\n"  # recipe-text 닫기
        response += "</div>\n"  # recipe-container 닫기
        responses.append(response)

    return '\n'.join(responses)


def recommend_recipes_by_ingredients_ml(model, tokenizer, ingredients, data):
    text = ' '.join(sorted(ingredients))
    X_text = prepare_text(text, tokenizer)

    predicted_class = model.predict(X_text).argmax(axis=1)[0]
    predicted_recipe_name = class_to_recipe_name[predicted_class]

    recommended_recipes = data[data['요리명'] == predicted_recipe_name]
    recommended_recipes = recommended_recipes.sort_values(by=['조회수', '스크랩수'], ascending=[False, False]).head(3)

    responses = []
    if not recommended_recipes.empty:
        for i, match in enumerate(recommended_recipes.iterrows(), start=1):
            response = "<div class='recipe-container'>\n"
            # 아이콘을 왼쪽에 추가합니다.
            response += f"<img class='recipe-icon' src='/static/images/icon1.png' alt='Recipe Icon'>\n"
            # 응답 박스를 오른쪽에 추가합니다.
            response += f"<div class='recipe-box'>\n"
            if i == 1:
                response += f"<div style='text-align: center; margin-bottom: 10px;'><p style='color: #4a83c9; font-weight: bold; font-size: 1.5em;'  class='cooking-partner'>쿠킹파트너 : {match[1]['요리명']} 레시피를 알려드리겠습니다.</p></div>\n"

            response += f"<p class='recipe-title'><strong>{i}. {match[1]['레시피제목']}</strong></p>\n"

            # 요리 재료 내용에서 줄바꿈 구현
            ingredients = match[1]['요리재료내용'].split('\n')
            formatted_ingredients = "<br>".join(ingredients)
            response += f"<p>{formatted_ingredients}</p>\n"

            # 조리순서를 구분자인 숫자와 마침표를 기준으로 나누어 줄바꿈합니다.
            cooking_steps = match[1]['조리순서'].split('\n')
            formatted_cooking_steps = "<br>".join(cooking_steps)
            response += f"<p><span class='cooking-step'>[조리순서]</span><br>{formatted_cooking_steps}</p>\n"
            response += "</div>\n"  # recipe-text 닫기
            response += "</div>\n"  # recipe-container 닫기
            responses.append(response)
    else:
        responses.append("<p class='no-recipe'>쿠킹파트너: 해당하는 레시피가 없습니다.</p>")

    return '\n'.join(responses), recommended_recipes
