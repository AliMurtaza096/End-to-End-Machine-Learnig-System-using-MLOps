import os
from flask import render_template,request,session,jsonify,send_file

from src.prediction import ChurnPredict
from image.prediction import DiseasePredict

from src.retrain import Retrain
from WebApp import app      
from .models import User_Details,User_DetailsSchema,db
from omegaconf import OmegaConf
import json
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime

# predictor = DiseasePredict("runs:/215fe8821da24c199208cb1c267ab88c/model")
# loaded_model = predictor.load_model()
user_details_schema = User_DetailsSchema()
user_details_List_schema = User_DetailsSchema(many=True)

cfg = OmegaConf.load('./config_dir/config.yaml')

app.secret_key = "asdsadsdvbvsdgvcgjsdvvsdcvg"

@app.route("/")
def index():
   
    return render_template('index.html')

@app.route("/signup", methods=["POST","GET"])
def signup():
    if request.method =="POST":
        print("IN POST")
        try:
            email =request.form['email']
                
            password = request.form['password']
            
            cpassword = request.form['cpassword']
            
            if password == cpassword:
                user_details_item  = User_Details(email,password)
                db.session.add(user_details_item)
                db.session.commit()
                return {"status":"OK"}
            else:
                return {"status":"Password and Confirm password Not same"}
        except:
            return {"status":"Email already exists"}
        
    return render_template('signup.html')


@app.route("/login", methods=["GET","POST"])
def login():
    """endpoint to show all todo items"""
    try:
        if request.method =="POST":

            user = User_Details.query.filter_by(email=request.form['email']).first()
            # print(user.email,user.password)
            # try:
            email = user.email
            
            session['loggedin'] = True
            session['email'] = email
        
            password= user.password
            
            if (email or password) == '':
                return  {"status" :"Either email or password  is not Valid"}
            elif password == request.form['password'] :
                return {"status": "OK"}
            else:

                return  {"status" :"Either email or password  is not Valid"}
    except:
        return {'status':"Invalid Email or Password"}

    return render_template('login.html')


@app.route("/home", methods = ['GET','POST'])
def home():
    return render_template('home.html')


@app.route("/dashboard",methods=['GET','POST'])
def dashboard():
    email = session['email']
    if request.method =='POST':
        if request.form['submit-button'] =='uploadDataset':
        
            dataset_file = request.files['file']
            filename = secure_filename(dataset_file.filename)
            dataset_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_data = pd.read_csv(file_path)
            
            predictor = ChurnPredict(cfg.churn_paths.model_artifact_dir,**{'data' :file_data})
            
            model_response = predictor.batch_predict()
            # df = pd.DataFrame(model_response)
            
            new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
            
            model_response.to_csv(os.path.join(cfg.churn_paths.save_file_path,new_filename))
            

            # model_response = list(model_response)
            # model_response =  [int(i) for i in model_response]
            
            # model_response = json.dumps({'status':model_response})

            # return jsonify(model_response)
            
            return send_file(os.path.join(cfg.churn_paths.save_file_path, new_filename))
        elif request.form['submit-button'] =='uploadTrainDataset':
            print('uploadTrainDataset')
            dataset_file = request.files['file']
            filename = secure_filename(dataset_file.filename)
            dataset_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(file_path)
            retrain = Retrain(file_path)
            retrain.retrain_model()
            
            # main.main(file_path)
            return {'status':'Model has been successfully trained on your data'}
    return render_template('dashboard.html',data= email)


# @app.route("/prediction/train")
# def training():
    

@app.route("/prediction", methods=["GET","POST"])
def prediction():
    if request.method == "POST":
        model_input = {
    
            
        'credit_score' :request.form['credit_score'],
        'geography' :request.form['geography'],
        'gender' : request.form['gender'],
        'age' : request.form['age'],
        'tenure': request.form['tenure'],
        'balance' : request.form['balance'],
        'num_of_products' : request.form['num_of_products'],
        'has_card':  request.form['has_credit_card'],
        'is_active_member': request.form['is_active_member'],
        'estimated_salary' : request.form['estimated_salary']
        
        
        }
        
        predictor = ChurnPredict(cfg.churn_paths.model_artifact_dir,**model_input)
        
        model_response = predictor.predict()
        model_response =model_response[0]
        model_response = json.dumps({'status':int(model_response)})
        return  jsonify(model_response)
        # except:
        #     return {"status":"Input Not Valid"}
    return render_template("prediction.html")

@app.route("/disease_predict",methods=['POST',"GET"])
def disease_predict():
    
    if request.method =="POST":
        
        # Get the uploaded file from the form
        uploaded_file = request.files['file']
    
        
        if uploaded_file:
            # Save the file to a folder on the server
            uploaded_file.save('E:/FYP/mlops/MLOps-FYP/WebApp/templates/static/uploads/' + uploaded_file.filename)
            # return
            prediction = predictor.predict(uploaded_file,loaded_model)
            # # print(prediction)
            # # # Render the uploaded image on the screen
            prediction = json.dumps({'status':prediction})
            return jsonify(prediction)
            

    return render_template('disease_predict.html')


@app.before_request
def make_session_permanent():
    session.permanent = True
            
        
        