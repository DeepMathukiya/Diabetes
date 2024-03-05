from flask import Flask, render_template, request
from joblib import load
import numpy as np

model = load("diabetes.joblib")
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict' , methods=['GET','POST'])
def onpredict():
    if request.method == 'POST':
        feature = []
        feature.append(request.form['Gender'])
        feature.append(request.form['Age'])
        feature.append(request.form['HYP'])
        feature.append(request.form['HD'])
        feature.append(request.form['SH'])
        feature.append(request.form['BMI'])
        feature.append(request.form['HbA'])
        feature.append(request.form['Glucose'])
    
        final = [np.array(feature)]
        op = model.predict(final)



    return render_template("index.html" , output= op)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port='7000')        
