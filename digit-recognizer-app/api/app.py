from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import logging

logging.basicConfig(
    level=logging.INFO,  # Or WARNING/ERROR/DEBUG
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


app = Flask(
    __name__,
    template_folder=os.path.abspath("../templates"),
    static_folder=os.path.abspath("../static")
)

@app.route('/')
def index():
    logger.info("Rendering home page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received a request to predict the digit")

    # Get model type from form data
    model_type = request.form.get('model_type', 'simple').lower()
    model_path = f"../model/mnist_{model_type}_model.h5"

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        return jsonify({'error': f'Could not load model: {e}'}), 400

    file = request.files['file']
    img = Image.open(file).convert('L').resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)[0]
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    logger.info(f"Prediction: {predicted_class}, Confidence: {confidence}")
    return jsonify({
        'digit': predicted_class,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
