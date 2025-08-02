from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("../model/mnist_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request to predict the digit")
    file = request.files['file']
    img = Image.open(file).convert('L').resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)[0]
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        'digit': predicted_class,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
