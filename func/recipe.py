import pandas as pd
import json


recipe = pd.read_excel('food/static/최종데이터프레임.xlsx')

def select_menu(food_name):
    temp = recipe[recipe['요리명'] == food_name]
    temp = temp.sort_values(by='조회수', ascending=False)
    temp['조리순서길이'] = temp['조리순서'].apply(len)

    return temp
#
# selected_menu = select_menu('감자조림')
# print(selected_menu)
#
#
#
# with open('D:/song/project/project3/web/food/model/food_classes.json', 'r', encoding='utf-8') as json_file:
#     food_classes = json.load(json_file)
# class_label = {v: k for k, v in food_classes.items()}
#
# selected_menu = select_menu(class_label[3])
# print(selected_menu)
#
# print(class_label[3])
# print(class_label[3] == '감자조림')
#
# import unicodedata
#
# label1 = class_label[3]
# label2 = '감자조림'
#
# label1_normalized = unicodedata.normalize('NFC', label1)
# label2_normalized = unicodedata.normalize('NFC', label2)
#
# print(label1_normalized == label2_normalized)
#
# selected_menu = select_menu(label1_normalized)
# print(selected_menu)
#
# label1_normalized = unicodedata.normalize('NFD', label1)
# label2_normalized = unicodedata.normalize('NFD', label2)
#
# print(label1_normalized == label2_normalized)

