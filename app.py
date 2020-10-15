from __future__ import division, print_function
# coding=utf-8
import sys
import numpy as np
import os
import glob
import re

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/weights_best.h5'

#Load the trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        img = image.load_img(file_path, target_size=(96,96))

        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)

        images = np.vstack([x])
        val = model.predict(images)

        str1 = 'Parasitized Cell'
        str2 = 'Uninfected'
        
        if val == 0:
            return str1
        else:
            return str2    
        
        os.remove(file_path)#removes file from the server after prediction has been returned
        
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
    app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
