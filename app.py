import tensorflow as tf
import keras
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello():
    return render_template("index.html")



@app.route("/",methods=['POST'])
def predict():
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    model = keras.models.load_model('E:\\Sem 8\\SCOA\\potatoes.h5')

    image = load_img(image_path, target_size=(256, 256))

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    # return predicted_class, confidence
    result = '%s (%.2f%%)' % (predicted_class,confidence)

    return render_template("index.html", prediction = result)

if __name__ == '__main__':
    app.run(port = 3000, debug=True)