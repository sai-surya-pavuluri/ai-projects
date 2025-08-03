import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [model, setModel] = useState('simple');
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setResult(null);
    setError(null);

    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    } else {
      setPreview(null);
    }
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('file', image);
    formData.append('model_type', model);

    if (!image) {
      setError("Please upload an image file.");
      return;
    }

    try {
      const res = await axios.post('http://localhost:5000/predict', formData);
      setResult(res.data);
    } catch (err) {
      console.error("Prediction failed", err);
      setError("Server is down or unreachable. Please try again later.");
      setResult(null);
    }
  };

  return (
    <div className="app-container">
      <div className="app-content">
        <h1>AI Digit Recognizer Tool</h1>

        <label style={{ fontWeight: 'bold' }}>CNN Model: </label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="simple">Simple</option>
          <option value="lenet">LeNet</option>
          <option value="resnet">ResNet</option>
          <option value="vgg">VGG</option>
        </select>

        <br /><br />
        <input
          type="file"
          onChange={handleFileChange}
          accept="image/*"
        />
        <button onClick={handleSubmit} disabled={!image}>Predict</button>

        {preview && (
          <div>
            <h3>Uploaded Image</h3>
            <img src={preview} alt="preview" width={100} height={100} />
          </div>
        )}

        {result && (
          <div>
            <h2>Prediction: {result.digit}</h2>
            <p>Confidence: {result.confidence}%</p>
          </div>
        )}

        {error && (
          <div style={{ color: 'red', marginTop: '1rem' }}>
            <strong>{error}</strong>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
