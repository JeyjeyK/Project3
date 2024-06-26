from flask import Flask

import os

# app 구동을 위한 부분
# 함수이름(create_app)과 return 부분의 이름은 그대로 사용
# def create_app(): 이 만들고자 하는 app의 실행과정을 설명하는
def create_app():

    # app의 이름
    # cmd 옵션에서 set FLASK_APP=food로 설정하여서
    # app 이름은 food로 설정됨
    app = Flask(__name__)

    # 사진을 담기위한 폴더 지정
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists('uploads'):
        os.makedirs('uploads')


    from .views import main_views, select_page_views
    from .views import text_search, image_search

    app.register_blueprint(main_views.bp)
    app.register_blueprint(select_page_views.bp)
    app.register_blueprint(text_search.bp)
    app.register_blueprint(image_search.bp)


    return app  # 변경 불가!