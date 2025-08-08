import React from 'react'
import { toast } from "react-toastify"

export default function DiaryForm({ emotion }) {
  const [diaryText, setDiaryText] = React.useState('')
  const [loading, setLoading] = React.useState(false)

  const handleChange = (e) => {
    setDiaryText(e.target.value)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!diaryText.trim()) {
      toast.warn("Please write something before saving.")
      return
    }

    try {
      setLoading(true)
      const res = await fetch('http://localhost:5000/diary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          emotion,
          entry: diaryText.trim()
        }),
      })

      const data = await res.json().catch(() => ({}))

      if (res.ok && data.status === "saved") {
        setDiaryText('')
        toast.success("Diary entry saved successfully!")
      } else {
        toast.error(data?.error || "Failed to save diary entry.")
      }
    } catch (err) {
      toast.error("Network error. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  if (!emotion) return null

  return (
    <div className="diary-form-card">
      <h2>Diary Entry for: {emotion}</h2>

      <form onSubmit={handleSubmit}>
        <textarea
          placeholder="Write about your thoughts..."
          rows={5}
          value={diaryText}
          onChange={handleChange}
          style={{ width: '100%', maxWidth: 500 }}
        />
        <br />
        <button type="submit" disabled={loading || !diaryText.trim()}>
          {loading ? "Saving..." : "Save Entry"}
        </button>
      </form>
    </div>
  )
}
