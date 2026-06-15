import { useState } from "react"
import "./App.css"

const DURATION_OPTIONS = [
  { value: "short", label: "Short", tooltip: "Under 4 minutes" },
  { value: "medium", label: "Medium", tooltip: "Between 4 and 20 minutes" },
  { value: "long", label: "Long", tooltip: "Over 20 minutes" },
]

const MIN_VIEWS_OPTIONS = [
  { value: "", label: "Any" },
  { value: "1000", label: "1K+" },
  { value: "10000", label: "10K+" },
  { value: "100000", label: "100K+" },
  { value: "1000000", label: "1M+" },
]

function App() {
  const [query, setQuery] = useState("")
  const [provider, setProvider] = useState("openai")
  const [status, setStatus] = useState("")
  const [videos, setVideos] = useState([])
  const [final, setFinal] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [activeVideoId, setActiveVideoId] = useState(null)

  // Advanced filters
  const [showFilters, setShowFilters] = useState(false)
  const [publishedAfterYear, setPublishedAfterYear] = useState(2010)
  const [duration, setDuration] = useState("")
  const [minViews, setMinViews] = useState("")
  const [maxVideos, setMaxVideos] = useState(5)

  async function handleSummarize() {
    setLoading(true)
    setError("")
    setVideos([])
    setFinal(null)
    setStatus("Starting...")
    setActiveVideoId(null)

    try {
      const body = {
        query,
        provider,
        max_videos: maxVideos,
        published_after_year: publishedAfterYear > 2010 ? publishedAfterYear : null,
        duration: duration || null,
        min_views: minViews ? parseInt(minViews) : null,
      }

      const response = await fetch("http://localhost:8000/summarize/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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

      {/* Advanced Filters */}
      <div className="filters-toggle" onClick={() => setShowFilters(!showFilters)}>
        {showFilters ? "▲" : "▼"} Advanced filters
      </div>

      {showFilters && (
        <div className="filters-panel">

          <div className="filter-group">
            <label>Published after: <strong>{publishedAfterYear === 2010 ? "Any year" : publishedAfterYear}</strong></label>
            <input
              type="range"
              min="2010"
              max="2026"
              value={publishedAfterYear}
              onChange={(e) => setPublishedAfterYear(parseInt(e.target.value))}
            />
            <div className="range-labels"><span>2010</span><span>2026</span></div>
          </div>

          <div className="filter-group">
            <label>Duration</label>
            <div className="radio-group">
              <label className="radio-option">
                <input type="radio" name="duration" value="" checked={duration === ""} onChange={() => setDuration("")} />
                Any
              </label>
              {DURATION_OPTIONS.map((opt) => (
                <label key={opt.value} className="radio-option" title={opt.tooltip}>
                  <input
                    type="radio"
                    name="duration"
                    value={opt.value}
                    checked={duration === opt.value}
                    onChange={() => setDuration(opt.value)}
                  />
                  {opt.label}
                  <span className="tooltip">{opt.tooltip}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <label>Minimum views</label>
            <select value={minViews} onChange={(e) => setMinViews(e.target.value)}>
              {MIN_VIEWS_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Number of videos: <strong>{maxVideos}</strong></label>
            <input
              type="range"
              min="3"
              max="20"
              value={maxVideos}
              onChange={(e) => setMaxVideos(parseInt(e.target.value))}
            />
            <div className="range-labels"><span>3</span><span>20</span></div>
          </div>

        </div>
      )}

      {error && <div className="error">{error}</div>}

      {status && (
        <div className="loading">
          <div className="spinner" />
          <p>{status}</p>
        </div>
      )}

      {hasResults && (
        <div className="results-layout">
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
