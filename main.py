from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

import sklearn
print(sklearn.__version__)
import sys
print(sys.executable)

app = Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
crop=pd.read_csv('crop-data.csv')


@app.route('/', methods=['GET','POST'])
def index():

    counties=sorted(crop['county'].unique())
    items = sorted(crop['item'].unique())
    seasons = sorted(crop['season'].unique())
    temps = sorted(crop['temp'].unique())
    rainfalls = sorted(crop['rainfall'].unique())
    phs = sorted(crop['ph'].unique())

    counties.insert(0,'Select county')
    items.insert(0, 'Select crop')
    seasons.insert(0, 'Select season')
    temps.insert(0, 'Select Tempreture')
    rainfalls.insert(0, 'Select rainfall')
    phs.insert(0, 'Select ph')
    return render_template('index.html', counties=counties,items=items,seasons=seasons,temps=temps,rainfalls=rainfalls,phs=phs)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    county = request.form.get('county')
    item = request.form.get('item')
    season = request.form.get('season')
    temp = request.form.get('temp')
    rainfall = request.form.get('rainfall')
    ph = request.form.get('ph')

    prediction = model.predict(pd.DataFrame(columns=['county','item','season','temp','rainfall','ph'], data=np.array([county,item,season,temp,rainfall,ph]).reshape(1,6)))

    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()