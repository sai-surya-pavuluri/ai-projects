import React, { useState } from 'react'
import './App.css'
import ImageUpload from './components/ImageUpload'
import EmotionResult from './components/EmotionResult'
import DiaryForm from './components/DiaryForm'
import { ToastContainer } from "react-toastify"
import "react-toastify/dist/ReactToastify.css"

function App() {
  const [predictedEmotion, setPredictedEmotion] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [showDiary, setShowDiary] = useState(false)

  const handlePrediction = (emotion, conf) => {
    setPredictedEmotion(emotion)
    setConfidence(conf)
    setShowDiary(false)
  }

  return (
    <div className="App">
      <h1>Emotion Diary</h1>

      <ImageUpload onPrediction={handlePrediction} />

      {predictedEmotion && confidence !== null && (
        <EmotionResult
          emotion={predictedEmotion}
          confidence={confidence}
          onShowDiary={() => setShowDiary(true)}
        />
      )}

      {showDiary && predictedEmotion && (
        <DiaryForm emotion={predictedEmotion} />
      )}

      <ToastContainer position="top-right" />
    </div>
  )
}

export default App
