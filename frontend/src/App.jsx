import { useState } from "react"
import "./App.css"

function App() {
  const [query, setQuery] = useState("")
  const [provider, setProvider] = useState("openai")
  const [status, setStatus] = useState("")
  const [videos, setVideos] = useState([])
  const [final, setFinal] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  async function handleSummarize() {
    setLoading(true)
    setError("")
    setVideos([])
    setFinal(null)
    setStatus("Starting...")

    try {
      const response = await fetch("http://localhost:8000/summarize/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, provider, max_videos: 5 }),
      })

      // Read the response as a stream of text chunks
      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // Each chunk may contain multiple newline-delimited JSON lines
        const lines = decoder.decode(value).split("\n").filter(Boolean)
        for (const line of lines) {
          const chunk = JSON.parse(line)

          if (chunk.type === "status") {
            setStatus(chunk.message)
          } else if (chunk.type === "video") {
            // Append each video as it arrives — React re-renders immediately
            setVideos((prev) => [...prev, chunk.data])
          } else if (chunk.type === "final") {
            setFinal(chunk.data)
            setStatus("")
          } else if (chunk.type === "error") {
            setError(chunk.message)
            setStatus("")
          }
        }
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setStatus("")
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

      {status && (
        <div className="loading">
          <div className="spinner" />
          <p>{status}</p>
        </div>
      )}

      {videos.length > 0 && (
        <section className="videos-section">
          <h2>Videos Analyzed</h2>
          {videos.map((v) => (
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
      )}

      {final && (
        <>
          <section className="summary-section">
            <h2>Overview</h2>
            <p>{final.final_summary}</p>
          </section>

          <section className="takeaways-section">
            <h2>Key Takeaways</h2>
            <ol>
              {final.key_takeaways.map((t, i) => (
                <li key={i}>{t}</li>
              ))}
            </ol>
          </section>
        </>
      )}
    </div>
  )
}

export default App
