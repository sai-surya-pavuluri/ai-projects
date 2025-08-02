import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);


  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImage(file);

    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setPreview(null);
    }
  };


  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('file', image);

    try {
      const res = await axios.post('http://localhost:5000/predict', formData);
      setResult(res.data);
    } catch (err) {
      console.error("Prediction failed", err);
    }
  };

  return (
    <div style={{ textAlign: 'center', marginTop: '2rem' }}>
      <h1>AI Digit Recognizer Tool</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleSubmit}>Predict</button>
      
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

    </div>
  );
}

export default App;
