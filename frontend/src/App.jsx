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
  const [activeVideoId, setActiveVideoId] = useState(null)

  async function handleSummarize() {
    setLoading(true)
    setError("")
    setVideos([])
    setFinal(null)
    setStatus("Starting...")
    setActiveVideoId(null)

    try {
      const response = await fetch("http://localhost:8000/summarize/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, provider, max_videos: 5 }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const lines = decoder.decode(value).split("\n").filter(Boolean)
        for (const line of lines) {
          const chunk = JSON.parse(line)

          if (chunk.type === "status") {
            setStatus(chunk.message)
          } else if (chunk.type === "video") {
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

  const hasResults = videos.length > 0 || final

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

      {hasResults && (
        <div className="results-layout">
          {/* Left column — video list + summary */}
          <div className="results-left">
            {videos.length > 0 && (
              <section className="videos-section">
                <h2>Videos Analyzed</h2>
                {videos.map((v) => (
                  <div key={v.video_id} className={`video-card ${activeVideoId === v.video_id ? "active" : ""}`}>
                    <h3>{v.title}</h3>
                    <p className="meta">{v.channel_name} · {v.view_count.toLocaleString()} views</p>
                    <p>{v.raw_summary}</p>
                    <ul>
                      {v.key_points.map((p, i) => (
                        <li key={i}>{p}</li>
                      ))}
                    </ul>
                    <button
                      className="watch-btn"
                      onClick={() => setActiveVideoId(v.video_id === activeVideoId ? null : v.video_id)}
                    >
                      {activeVideoId === v.video_id ? "▼ Close" : "▶ Watch Video"}
                    </button>
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

          {/* Right column — video player (sticky) */}
          {activeVideoId && (
            <div className="player-panel">
              <iframe
                src={`https://www.youtube.com/embed/${activeVideoId}`}
                title="YouTube video player"
                allowFullScreen
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              />
              <button className="close-btn" onClick={() => setActiveVideoId(null)}>✕ Close</button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
