from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import PIL

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    
    cnn=load_model('D:\Works\def\model\dl_model_2.h5')
    
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    # Predict probabilities for each class
    probabilities = cnn.predict(test_image)
    # Get the index of the class with the highest probability
    Categories = ['Fire', 'Smoke']
    ans = int(probabilities)
    if ans == 0:
        classification = "Fire"
    else:
        classification = "Smoke"
    
    return render_template('home.html', prediction=classification)


if __name__ == '__main__':
    app.run(port = 3000, debug=True)