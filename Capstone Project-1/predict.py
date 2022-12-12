
import numpy as np
import tensorflow as tf 


from tensorflow import keras 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet import preprocess_input


from flask import Flask
from flask import request
from flask import jsonify

model_file = final_model.h5
model = keras.models.load_model(model_file)

classes = [
    'Cabernet Sauvignon',
    'Muller Thurgau',
    'Auxerrois',
    'Syrah',
    'Sauvignon Blanc',
    'Tempranillo',
    'Riesling',
    'Pinot Noir',
    'Chardonnay',
    'Cabernet Franc',
    'Merlot'
    ]

app = Flask('response')

@app.route('/predict', methods=['POST'])

def predict():
    filepath = request.get_json()
    img = load_img(filepath, target_size=(256,256))
    x = np.array(img)
    X = np.array([x])
    X = preprocess_input(X)
    pred = model.predict(X).tolist()
    result = dict(zip(classes,pred[0]))
    return jsonify(result)


if __name__ == "_main_":
    app.run(debug=True, host='0.0.0.0',port=8888)
    
    