from flask import Flask, render_template, request, redirect
from details import models,categorical_columns
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html',models=models)

@app.route('/predictor/<name>',methods=['GET','POST'])
def predictors(name):
    if request.method=='POST':
        arr = []
        for attribute in models[name].attributes:
            arr.append((float)(request.form[attribute]))

        loaded_model = pickle.load(open(f"./prediction_models/{name}_model.sav",'rb'))
        input_data = np.asarray(arr)
        input_data_reshaped = input_data.reshape(1,-1)
        prediction = loaded_model.predict(input_data_reshaped)
        return render_template('form.html',name=name,models=models,prediction=prediction[0],categorical_columns=categorical_columns)

    return render_template('form.html',name=name,models=models,prediction=-1,categorical_columns=categorical_columns)

@app.route('/<name>')
def nav_links(name):
    return render_template(f"{name}.html")

if __name__ == '__main__':
    app.run(debug=True,port=8000)
