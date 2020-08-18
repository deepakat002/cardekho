from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn  #here sklearn=0.22 version is installed. v0.31 gives error
app = Flask(__name__)

#make sure you have installed gunicorn in this env

#loading pickle files
model = pickle.load(open('rf_final.pkl', 'rb'))
transformer = pickle.load(open('kms_present_transformer.pkl','rb'))
encoder = pickle.load(open('leave_one_out_encoder.pkl','rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        car_name = request.form['vehicle_name']
        d = pd.DataFrame({'Car_Name':[car_name]})
        
        #Encoding car name
        encode = encoder.transform(d['Car_Name'])
        car_name = encode['Car_Name']
        
        #year
        Year = int(request.form['Year'])
        Age=2020-Year
        
        
        Present_Price=float(request.form['Present_Price'])      
        Kms_Driven=int(request.form['Kms_Driven'])
        
        #tranforming the kms and present value
        transform = transformer.transform([[Kms_Driven,Present_Price]])
        Kms_Driven = transform[0][0]
        Present_Price = transform[0][1]
        
        
        Owner=int(request.form['Owner'])
        Fuel_Type_Diesel=0
        Fuel_Type=request.form['Fuel_Type']
        if(Fuel_Type=='Petrol'):
                Fuel_Type_Petrol=1
                Fuel_Type_Diesel=0
        elif(Fuel_Type == 'CNG'):
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=0        
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1

        Seller_Type=request.form['Seller_Type']
        if(Seller_Type=='Individual'):
            Seller_Type_Individual=1
        else:
            Seller_Type_Individual=0	
        Transmission =request.form['Transmission']
        if(Transmission =='Mannual'):
            Transmission_Mannual=1
        else:
            Transmission_Mannual=0
        prediction=model.predict([[car_name,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual,Kms_Driven,Present_Price,Owner,Age]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry!! you cannot sell this vehicle.")
        else:
            return render_template('index.html',prediction_text="You can sell this vehicle at {} lakhs".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)