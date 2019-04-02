import os
import tensorflow as tf
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
import numpy as np
import pandas as pd

from flask import Flask, request, redirect, url_for, jsonify

from keras.models import load_model
from keras import backend as K


model = load_model("ECG_DNN_trained.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'

graph = K.get_session().graph

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # ensuring the folder location exists 
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(filepath)

            #import the CSV into a pandas DataFrame
            data_df = pd.read_csv(filepath, low_memory=False, header=None)

            #make predictions
            global graph
            with graph.as_default():
                results = model.predict_proba(data_df)
                
                #format predictions
                Normal_results="{:.9f}".format(float(results[0][0]))
                SP_results="{:.9f}".format(float(results[0][1]))
                PVC_results="{:.9f}".format(float(results[0][2]))
                FV_results="{:.9f}".format(float(results[0][3]))

                #print all results 
                return (f"Normal %: {Normal_results} / Supraventricular Premature Beat %: {SP_results} / Premature Ventricular Contraction %: {PVC_results} / Fusion of Ventricular and Normal Beat %: {FV_results}")

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new EKG CSV (1 row x 187 columns)</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)