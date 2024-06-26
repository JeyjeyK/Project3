from flask import Blueprint, render_template, request, redirect, current_app, url_for, jsonify

bp = Blueprint('select', __name__, url_prefix='/main')


@bp.route('/', methods = ['GET', 'POST'])
def select_page():

   return render_template('main/main.html')