import { useState } from "react"
import "./App.css"

const DURATION_OPTIONS = [
  { value: "short", label: "Short", tooltip: "Under 4 minutes" },
  { value: "medium", label: "Medium", tooltip: "4–20 minutes" },
  { value: "long", label: "Long", tooltip: "Over 20 minutes" },
]

const MIN_VIEWS_OPTIONS = [
  { value: "", label: "Any views" },
  { value: "1000", label: "1K+" },
  { value: "10000", label: "10K+" },
  { value: "100000", label: "100K+" },
  { value: "1000000", label: "1M+" },
]

function formatViews(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M"
  if (n >= 1_000) return (n / 1_000).toFixed(0) + "K"
  return n.toString()
}

function App() {
  const [query, setQuery] = useState("")
  const [provider, setProvider] = useState("openai")
  const [status, setStatus] = useState("")
  const [videos, setVideos] = useState([])
  const [final, setFinal] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [activeVideoId, setActiveVideoId] = useState(null)
  const [questions, setQuestions] = useState({})
  const [answers, setAnswers] = useState({})
  const [askingVideoId, setAskingVideoId] = useState(null)

  const [mode, setMode] = useState("search")
  const [urlInput, setUrlInput] = useState("")

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
    setAnswers({})

    try {
      const isUrlMode = mode === "urls"
      const endpoint = isUrlMode ? "http://localhost:8000/summarize-urls/stream" : "http://localhost:8000/summarize/stream"
      const body = isUrlMode
        ? { urls: urlInput.split("\n").map(u => u.trim()).filter(Boolean), provider }
        : {
            query,
            provider,
            max_videos: maxVideos,
            published_after_year: publishedAfterYear > 2010 ? publishedAfterYear : null,
            duration: duration || null,
            min_views: minViews ? parseInt(minViews) : null,
          }

      const response = await fetch(endpoint, {
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
          if (chunk.type === "status") setStatus(chunk.message)
          else if (chunk.type === "video") setVideos((prev) => [...prev, chunk.data])
          else if (chunk.type === "final") { setFinal(chunk.data); setStatus("") }
          else if (chunk.type === "error") { setError(chunk.message); setStatus("") }
        }
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setStatus("")
    }
  }

  async function handleAsk(video) {
    const question = questions[video.video_id]
    if (!question?.trim()) return
    setAskingVideoId(video.video_id)
    setQuestions((prev) => ({ ...prev, [video.video_id]: "" }))

    setAnswers((prev) => ({
      ...prev,
      [video.video_id]: [...(prev[video.video_id] || []), { question, answer: "" }],
    }))

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: video.video_id,
          question,
        }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const token = decoder.decode(value)
        setAnswers((prev) => {
          const history = [...(prev[video.video_id] || [])]
          const last = history[history.length - 1]
          history[history.length - 1] = { ...last, answer: last.answer + token }
          return { ...prev, [video.video_id]: history }
        })
      }
    } catch (err) {
      setAnswers((prev) => {
        const history = [...(prev[video.video_id] || [])]
        history[history.length - 1] = { ...history[history.length - 1], answer: "Error: " + err.message }
        return { ...prev, [video.video_id]: history }
      })
    } finally {
      setAskingVideoId(null)
    }
  }

  const hasResults = videos.length > 0 || final

  return (
    <div className="app">
      {/* Hero */}
      <div className="hero">
        <div className="hero-logo">
          <div className="hero-logo-icon">▶</div>
        </div>
        <h1>YouTube <span>Summarizer</span></h1>
        <p>Search any topic — get AI summaries, insights, and Q&A from the top videos</p>
      </div>

      {/* Tabs + Search */}
      <div className="search-container">
        <div className="mode-tabs">
          <button className={`mode-tab ${mode === "search" ? "active" : ""}`} onClick={() => setMode("search")}>
            🔍 Search by topic
          </button>
          <button className={`mode-tab ${mode === "urls" ? "active" : ""}`} onClick={() => setMode("urls")}>
            🔗 Analyze URLs
          </button>
        </div>

        {mode === "search" ? (
          <div className="search-bar">
            <input
              type="text"
              placeholder="e.g. machine learning, climate change, Chicago..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSummarize()}
            />
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="openai">OpenAI</option>
              <option value="claude">Claude</option>
            </select>
            <button onClick={handleSummarize} disabled={loading || !query.trim()}>
              {loading ? "Analyzing..." : "Summarize →"}
            </button>
          </div>
        ) : (
          <div className="url-input-container">
            <textarea
              placeholder={"Paste YouTube URLs, one per line:\nhttps://www.youtube.com/watch?v=...\nhttps://youtu.be/..."}
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              rows={4}
            />
            <div className="url-input-footer">
              <select value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="openai">OpenAI</option>
                <option value="claude">Claude</option>
              </select>
              <button onClick={handleSummarize} disabled={loading || !urlInput.trim()}>
                {loading ? "Analyzing..." : "Analyze Videos →"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Filters — only in search mode */}
      {mode === "search" && <div className="filters-toggle" onClick={() => setShowFilters(!showFilters)}>
        {showFilters ? "▲" : "▼"} Advanced filters
      </div>}

      {showFilters && mode === "search" && (
        <div className="filters-panel">
          <div className="filter-group">
            <label>Published after: {publishedAfterYear === 2010 ? "Any" : publishedAfterYear}</label>
            <input type="range" min="2010" max="2026" value={publishedAfterYear}
              onChange={(e) => setPublishedAfterYear(parseInt(e.target.value))} />
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
                  <input type="radio" name="duration" value={opt.value}
                    checked={duration === opt.value} onChange={() => setDuration(opt.value)} />
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
            <label>Videos to analyze: {maxVideos}</label>
            <input type="range" min="3" max="20" value={maxVideos}
              onChange={(e) => setMaxVideos(parseInt(e.target.value))} />
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
                <div className="section-title">Videos Analyzed</div>
                {videos.map((v, idx) => (
                  <div key={v.video_id} className={`video-card ${activeVideoId === v.video_id ? "is-active" : ""}`}>
                    <div className="video-card-header">
                      <div className="video-number">{idx + 1}</div>
                      <div className="video-card-header-text">
                        <h3>{v.title}</h3>
                        <div className="meta">
                          <span>{v.channel_name}</span>
                          <span className="meta-dot" />
                          <span>{formatViews(v.view_count)} views</span>
                        </div>
                      </div>
                    </div>

                    <p>{v.raw_summary}</p>

                    <ul>
                      {(v.key_points_timed?.length ? v.key_points_timed : v.key_points.map(text => ({ text, timestamp: null }))).map((kp, i) => (
                        <li key={i}>
                          <span>{kp.text}</span>
                          {kp.timestamp != null && (
                            <button className="timestamp-btn" onClick={() => {
                              setActiveVideoId(v.video_id)
                              document.querySelector(".player-panel iframe")?.setAttribute(
                                "src",
                                `https://www.youtube.com/embed/${v.video_id}?start=${kp.timestamp}`
                              )
                            }}>
                              {Math.floor(kp.timestamp / 60)}:{String(kp.timestamp % 60).padStart(2, "0")}
                            </button>
                          )}
                        </li>
                      ))}
                    </ul>

                    {v.categories?.length > 0 && (
                      <div className="category-chart">
                        <h4>Content Breakdown</h4>
                        {v.categories.map((c, i) => (
                          <div key={i} className="category-row">
                            <span className="category-label">{c.category}</span>
                            <div className="category-bar-bg">
                              <div className="category-bar-fill" style={{ width: `${c.percentage}%`, "--idx": i }} />
                            </div>
                            <span className="category-pct">{c.percentage}%</span>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="video-card-actions">
                      <button
                        className={`watch-btn ${activeVideoId === v.video_id ? "active" : ""}`}
                        onClick={() => setActiveVideoId(v.video_id === activeVideoId ? null : v.video_id)}
                      >
                        {activeVideoId === v.video_id ? "▼ Close Player" : "▶ Watch Video"}
                      </button>
                    </div>

                    <div className="qa-section">
                      <h4>Ask about this video</h4>
                      {answers[v.video_id]?.length > 0 && (
                        <div className="qa-history">
                          {answers[v.video_id].map((entry, i) => (
                            <div key={i} className="qa-entry">
                              <div className="qa-question">{entry.question}</div>
                              <div className="qa-answer">{entry.answer}</div>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="qa-input-row">
                        <input
                          type="text"
                          placeholder="Ask anything about this video..."
                          value={questions[v.video_id] || ""}
                          onChange={(e) => setQuestions((prev) => ({ ...prev, [v.video_id]: e.target.value }))}
                          onKeyDown={(e) => e.key === "Enter" && handleAsk(v)}
                        />
                        <button onClick={() => handleAsk(v)} disabled={askingVideoId === v.video_id}>
                          {askingVideoId === v.video_id ? "..." : "Ask"}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </section>
            )}

            {final && (
              <>
                <section className="summary-section">
                  <div className="section-title">Overview</div>
                  <p>{final.final_summary}</p>
                </section>
                <section className="takeaways-section">
                  <div className="section-title">Key Takeaways</div>
                  <ol>
                    {final.key_takeaways.map((t, i) => <li key={i}>{t}</li>)}
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
              <button className="close-btn" onClick={() => setActiveVideoId(null)}>✕ Close Player</button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
