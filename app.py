from flask import Flask, render_template, request, redirect, flash, url_for
import os
from main import getPrediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'C:/Users/zheng/OneDrive/Documents/PlayingProjects/CovidPneumonia/static/', filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(file_path)
        print(filename)
        output = getPrediction(file_path)
        print(output)
        
    return render_template('predict.html', prediction = output, user_image = '/static/'+filename, file_name=filename)  

if __name__ == "__main__":
    app.run()