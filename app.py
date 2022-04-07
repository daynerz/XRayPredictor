from flask import Flask, render_template, request, redirect, flash, url_for
import os
from main import getPrediction
from keys import staticFolder_path

app = Flask(__name__)
folder_path = staticFolder_path()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(folder_path, filename) 
        file.save(file_path)
        print(file_path)
        print(filename)
        output = getPrediction(file_path)
        print(output)
    return render_template('predict.html', prediction = output, user_image = '/static/'+filename, file_name=filename)  

if __name__ == "__main__":
    app.run()