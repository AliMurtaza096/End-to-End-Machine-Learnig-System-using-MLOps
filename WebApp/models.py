from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy

from WebApp import app

db = SQLAlchemy(app)
ma = Marshmallow(app)

class User_Details(db.Model):
    email = db.Column(db.String(120),primary_key=True)
    password = db.Column(db.String(120))

    def __init__(self, email, password):
        self.email = email
        self.password = password


class User_DetailsSchema(ma.Schema):
    class Meta:
        fields = ( 'email', 'password')
        
class User_Prediction_Data(db.Model):
    email= db.Column(db.String(120),primary_key=True)
    credit_score = db.Column(db.Integer)
    geography = db.Column(db.String(120))
    gender = db.Column(db.String(120))
    age  = db.Column(db.Integer)
    tenure = db.Column(db.Integer)
    balance = db.Column(db.Float)
    num_of_products = db.Column(db.Integer)
    has_card = db.Column(db.Integer)
    is_active_member = db.Column(db.Integer)
    estimated_salary = db.Column(db.Float)

    def __init__(self,email,**kwargs):
        self.email=email
        self.credit_score= kwargs['credit_score']
        self.geography = kwargs['geography']
        self.gender = kwargs['gender']
        self.age = kwargs['age']
        self.tenure = kwargs['tenure']
        self.balance = kwargs['balance']
        self.num_of_products = kwargs['num_of_products']
        self.has_card = kwargs['has_card']
        self.is_active_member =kwargs['is_active_member']
        self.estimated_salary = kwargs['estimated_salary']
        
    
class User_DetailsSchema(ma.Schema):
    class Meta:
        fields = ('email','credit_score','geography','gender',
                  'age','tenure','balance','has_card',
                  'num_of_products','is_active_member',
                  'estimated_salary')

db.create_all()