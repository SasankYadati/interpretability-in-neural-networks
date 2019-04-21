from flask import Flask
from flask import render_template
from flask import request
from flask import redirect, url_for
from flask import jsonify
app = Flask(__name__,static_url_path='/static')

import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.misc import toimage
import numpy as np
from PIL import Image
from static.clustering1 import *

def buildDataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.array([(i/255).reshape(28,28,1) for i in x_train])
    x_test = np.array([(i/255).reshape(28,28,1) for i in x_test])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np.array([tf.keras.utils.to_categorical(i,num_classes = 10).reshape(10,1) for i in y_train])
    y_test = np.array([tf.keras.utils.to_categorical(i,num_classes = 10).reshape(10,1) for i in y_test])

    return x_train, y_train, x_test, y_test

x_train,y_train,x_test,y_test = buildDataset()

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/api/click', methods=['POST'])
def getImage():
    idx = request.form["index"]
    print(idx)
    idx = int(idx)
    image = x_test[idx]
    filename = "static/test"+str(idx)+".jpg"
    file = "test"+str(idx)+".jpg"
    im = toimage(image.reshape(28,28))
    im.save(filename)
    return str(file)

@app.route('/api/output', methods=['POST'])
def getOutputs():
    idx = request.form["index"]
    print(idx)
    idx = int(idx)
    image = x_test[idx]

    l = getPDF(image.reshape(1,28,28,1),idx)
    encoding_test,original_image_filename,noisy_image_filename,output_scores_filename,concepts_image_filename,encodings_noisy_and_original_filename,theta_diff_filename,grounding_elements_filename,output_class,mse_theta,mse_encoded = l
    print(l)

    concept_activated = np.argmax(encoding_test)+1

    op = {'concept_activated' : str(concept_activated), 'image_file_name' : original_image_filename, 'noisy_image_file_name' : noisy_image_filename, 'output_scores_file_name': output_scores_filename,
    'concepts_image_file_name' : concepts_image_filename, 'encodings' : encodings_noisy_and_original_filename,'theta_diff': theta_diff_filename, 'grounding' : grounding_elements_filename,
    'output_class' : str(output_class), 'mse_theta' : str(mse_theta),'mse_encoded' : str(mse_encoded)}
    return jsonify(op)
