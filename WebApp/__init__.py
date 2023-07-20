import os

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
with app.app_context():
    CORS(app)

 
 
    UPLOAD_FOLDER = 'download'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    app.config['SQLALCHEMY_DATABASE_URI']= 'mysql+pymysql://admin:Sardar12@newmlopsdb.chs6rlz4ojkg.ap-south-1.rds.amazonaws.com:3306/newmlopsdb'
    app.config['UPLOAD_FOLDER']  =UPLOAD_FOLDER

    from WebApp import models
    from WebApp import views
    