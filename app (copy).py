from flask import Flask, render_template,request
import os
from joblib import load
import base64
import re
from PIL import Image
from io import BytesIO

import numpy as np
from scipy.misc import imsave, imread, imresize
from keras.models import model_from_json
import tensorflow as tf
import pandas as pd


maps = pd.read_csv("map.txt", delimiter = ' ', \
                   index_col=0, header=None, squeeze=True)

global model,graph

def load_model():
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model2.h5")
    print("Loaded Model from disk")
    
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    graph = tf.get_default_graph()
    
    return loaded_model,graph
    
model,graph = load_model()

app = Flask(__name__)

#decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)',imgData1.decode('utf-8')).group(1)
    #print(base64.b64decode(imgstr))
    byte_data = base64.b64decode(imgstr)
    print(byte_data)
    return
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    img.save('output.png',"PNG")
    return 
    with open('output.png','wb') as output:
        output.write(decode('base64'))

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    #imgData = request.get_data()
    data_url = request.get_data()
    convertImage(data_url)
    #print(imgData)
    x = imread('output.png',mode='L')
    #compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    x= x.astype('float32')
    x /= 255
    x = imresize(x,(28,28))
    x = x/255
    #imshow(x)
    #convert to a 4D tensor to feed into our model
    x = x.reshape(1,28,28,1)
    
    with graph.as_default():
        res = model.predict(x)
        out = np.argmax(res,axis=1)
        return(chr(maps[out]))
        
    

@app.route('/se/')
def sindex():
	#initModel()
	#render out pre-built HTML file right on the index page
	return 'secHai'


if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)