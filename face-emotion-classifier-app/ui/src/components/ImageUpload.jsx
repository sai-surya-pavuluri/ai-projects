import React from 'react'
import { toast } from "react-toastify"

export default function ImageUpload({ onPrediction }) {
  const [file, setFile] = React.useState(null)
  const [previewUrl, setPreviewUrl] = React.useState(null)
  const [loading, setLoading] = React.useState(false)

  const handleImageChange = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
  }

  const handleSubmit = async () => {
    if (!file) {
      toast.warn("Please select an image first.")
      return
    }

    try {
      setLoading(true)
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData
      })

      const data = await res.json().catch(() => ({}))

      if (res.ok) {
        const { emotion, confidence } = data
        onPrediction?.(emotion, confidence)
        toast.success("Prediction received!")
      } else {
        toast.error(data?.error || "Prediction failed.")
      }
    } catch (err) {
      toast.error("Network error. Is the Flask server running?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="upload-card">
      <h2>Upload Your Face</h2>

      <input type="file" accept="image/*" onChange={handleImageChange} />

      {previewUrl && (
        <div className="preview-box" style={{ marginTop: 12 }}>
          <img
            src={previewUrl}
            alt="preview"
            style={{ maxWidth: 300, borderRadius: 8 }}
          />
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        style={{ marginTop: 12 }}
      >
        {loading ? "Analyzing..." : "Analyze Emotion"}
      </button>
    </div>
  )
}