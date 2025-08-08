import React from 'react'

export default function EmotionResult({ emotion, confidence, onShowDiary }) {
  if (!emotion) return null

  const emojiMap = {
    happy: '😄',
    sad: '😢',
    angry: '😠',
    surprise: '😲',
    fear: '😨',
    disgust: '🤢',
    neutral: '😐',
  }

  const emoji = emojiMap[emotion.toLowerCase()] || '🤔'
  const confidencePercent = (typeof confidence === 'number')
    ? `${Math.round(confidence * 100)}%`
    : 'N/A'

  const shouldSuggestDiary =
    ['sad', 'angry', 'fear'].includes(emotion.toLowerCase())

  return (
    <div className={`emotion-result-card emotion-${emotion.toLowerCase()}`}>
      <h2>Your Emotion</h2>
      <p><strong>Emotion:</strong> {emotion} <span aria-label={emotion}>{emoji}</span></p>
      <p><strong>Confidence:</strong> {confidencePercent}</p>

      {shouldSuggestDiary && (
        <>
          <p>Would you like to write about how you're feeling?</p>
          <button onClick={onShowDiary}>Write Diary Entry</button>
        </>
      )}
    </div>
  )
}