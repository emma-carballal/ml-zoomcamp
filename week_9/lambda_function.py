import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import os
import tflite_runtime.interpreter as tflite

MODEL_NAME = os.getenv('MODEL_NAME', 'dino-vs-dragon-v2.tflite')

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg"

def predict(url):
    img = download_image(url)
    resized_img = prepare_image(img, (150, 150))
    x = np.array(resized_img, dtype='float32')
    X = np.array([x])
    X /= 255.0
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    return pred[0].tolist()

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {'prediction': pred}
    return result
