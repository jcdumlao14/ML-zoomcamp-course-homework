#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


create_preprocessor = create_preprocessor('mobilenet', target_size=(256,256))

interpreter = tflite.Interpreter(model_path='final_model.h5')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

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


def predict(url):
    X = preprocessor.from_url(url)
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    
    preds = interpreter.get_tensor(output_index)
    
    float_predictions = preds[0].tolist()
    
    return dict(zip(classes,float_predictions))


def lambda_handler(event, countext):
    url = event['url']
    result = predict(url)
    return result    