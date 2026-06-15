import { useState } from "react"
import "./App.css"

function App() {
  const [query, setQuery] = useState("")
  const [provider, setProvider] = useState("openai")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  async function handleSummarize() {
    setLoading(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("http://localhost:8000/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, provider, max_videos: 5 }),
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || "Server error")
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      // finally always runs — clears the spinner whether it succeeded or failed
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>YouTube Summarizer</h1>
        <p>Search any topic and get an AI-generated summary from the top YouTube videos</p>
      </header>

      <div className="search-bar">
        <input
          type="text"
          placeholder="e.g. machine learning, react hooks, climate change..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSummarize()}
        />

        <select value={provider} onChange={(e) => setProvider(e.target.value)}>
          <option value="openai">OpenAI</option>
          <option value="claude">Claude</option>
        </select>

        <button onClick={handleSummarize} disabled={loading || !query.trim()}>
          {loading ? "Summarizing..." : "Summarize"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Fetching and summarizing YouTube videos...</p>
        </div>
      )}

      {result && (
        <div className="results">
          <section className="summary-section">
            <h2>Overview</h2>
            <p>{result.final_summary}</p>
          </section>

          <section className="takeaways-section">
            <h2>Key Takeaways</h2>
            <ol>
              {result.key_takeaways.map((t, i) => (
                <li key={i}>{t}</li>
              ))}
            </ol>
          </section>

          <section className="videos-section">
            <h2>Videos Analyzed</h2>
            {result.videos.map((v) => (
              <div key={v.video_id} className="video-card">
                <a href={v.video_url} target="_blank" rel="noreferrer">
                  <h3>{v.title}</h3>
                </a>
                <p className="meta">{v.channel_name} · {v.view_count.toLocaleString()} views</p>
                <p>{v.raw_summary}</p>
                <ul>
                  {v.key_points.map((p, i) => (
                    <li key={i}>{p}</li>
                  ))}
                </ul>
              </div>
            ))}
          </section>
        </div>
      )}
    </div>
  )
}

export default App
