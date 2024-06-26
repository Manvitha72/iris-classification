from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import joblib as joblib
import os

# Load the trained model using pickle
with open('final_iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler using joblib
scaler = joblib.load('scaler.save')

app = Flask(__name__)

image_folder = os.path.join('static', 'images')
app.config['upload_folder'] = image_folder

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        sl = request.form['SepalLength']
        sw = request.form['SepalWidth']
        pl = request.form['PetalLength']
        pw = request.form['PetalWidth']
        data = pd.DataFrame([[sl, sw, pl, pw]], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
        print(f"Input data: {data}")

        x = scaler.transform(data)
        print(f"Transformed data: {x}")

        prediction = model.predict(x)
        print(f"Prediction: {prediction}")

        image = prediction[0] + '.png'
        image = os.path.join(app.config['upload_folder'], image)
        print(f"Image path: {image}")

        return render_template('index.html', prediction=prediction[0], image=image)
    return render_template('index.html', prediction=None, image=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
