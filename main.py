from flask import Flask
from flask import render_template
from flask import request
import numpy as np
import PIL.Image
import urllib.request
import os
from flask import flash, redirect, render_template
from werkzeug.utils import secure_filename
from nltk import FreqDist
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report
app = Flask(__name__)

UPLOAD_FOLDER = 'temp/'
app.secret_key = "secret key"
kumpulanTxt=[]
namaDokument=[]
listbaru=[]


@app.route('/')
def hello_world():
    return render_template('home.html')


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/data1', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            namaDokument.append(filename)
            print("INI FILE NAME",filename)
            try:
                file.save(os.path.join(UPLOAD_FOLDER, "dat.jpg"))
                cropgambar()
                hoho=predict()
                return render_template('search2.html',hasil=hoho)
            except IOError as e:
                print ('Operation failed: %s' % e.strerror)
            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


def cropgambar():
    image = PIL.Image.open("temp/dat.jpg")
    center=center_image(image)
    left,top,right,bottom = center
    center_cropped = crop(image,left,top,right,bottom)
    center_cropped.save("uploads/daging.jpeg")
    flash('Image successfully uploaded and displayed')
    

def predict():
    json_file = open('model50epochbaru.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model50epochbaru.h5")
    print("Loaded model from disk")
    opt2 = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0) 
    hehe="error"
    # evaluate loaded model on test data
    loaded_model.compile(optimizer = opt2, loss = 'binary_crossentropy', metrics = ['accuracy'])
    haha=[]
    count_pork = 0
    count_beef = 0
    test_image = image.load_img('uploads/daging.jpeg', target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    if result[0][0] == 0:
        prediction = 'pork'
        count_pork = count_pork + 1
    else:
        prediction = 'beef'
        count_beef = count_beef + 1
    
    print(haha)
    print("count_beef:" + str(count_beef))   
    print("count_pork:" + str(count_pork))
    if count_beef != 0:
        hehe="beef"
    elif count_pork != 0:
        hehe="pork"
    return hehe


def center_image(image):
    width, height = image.size
    left= width / 4
    top = height / 4
    right = 3 * width / 4
    bottom = 3 * height / 4
    return ((left,top,right,bottom))


def crop(image,left,top,right,bottom):
    cropped= image.crop((left,top,right,bottom))
    return cropped



