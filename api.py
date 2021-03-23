from flask import Flask, request, Response, jsonify
import jsonpickle
import numpy as np
import cv2
import time

from keras.models import load_model
import tensorflow as tf
graph = tf.get_default_graph()
model = load_model('data/pretrained/_vgg16_.52-0.93.hdf5', compile=False)

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/')
def index():
    response = {'status' : 'ok'}
    response = jsonify(response)

    return response

@app.route('/check_server',methods=['POST','GET'])
def check_server():
    data = request.form['image']
    print(data)
    response = {'status' : 'ok','data': data}
    response = jsonify(response)

    return response

# route http posts to this method
@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['image'].read()
    npimg = np.fromstring(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    #make prediction
    prediction = recognize(img)
    label = ['Alzheimer', 'Non Alzheimer']
    confidence = "%.2f" % (prediction[np.argmax(prediction)] * 100)
    pred_label = label[np.argmax(prediction)]

    response = {'label':pred_label,'acc' : confidence}
    response = jsonify(response)

    return response

def recognize(img):
    img = cv2.resize(img,(150,150))
    x = (img/225.)
    with graph.as_default():
        prediction = model.predict(np.expand_dims(x, axis=0))[0]

    return prediction

# start flask app
app.run(host="127.0.0.1", port=5000)