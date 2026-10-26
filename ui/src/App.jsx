import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('What are the key advantages of running LLM inference locally on consumer hardware?')
  const [maxTokens, setMaxTokens] = useState(150)
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [mode, setMode] = useState('real')
  const [streamingText, setStreamingText] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [metrics, setMetrics] = useState({ ttft: null, tps: null, time: null })
  const [telemetry, setTelemetry] = useState(null)
  
  const endRef = useRef(null)

  const fetchTelemetry = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8002/v1/telemetry')
      if (res.ok) {
        const data = await res.json()
        setTelemetry(data)
      }
    } catch {
      // Server not reachable yet
    }
  }

  useEffect(() => {
    fetchTelemetry()
    const interval = setInterval(fetchTelemetry, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [streamingText])

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true)
    setStreamingText('')
    setMetrics({ ttft: null, tps: null, time: null })
    
    const startTime = performance.now()
    let firstTokenTime = null
    let tokenCount = 0

    try {
      const response = await fetch('http://127.0.0.1:8002/v1/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          prompt, 
          max_tokens: parseInt(maxTokens), 
          temperature: parseFloat(temperature),
          top_p: parseFloat(topP),
          mode 
        })
      })
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let done = false

      while (!done) {
        const { value, done: readerDone } = await reader.read()
        done = readerDone
        if (value) {
          if (!firstTokenTime) {
            firstTokenTime = performance.now()
            setMetrics(prev => ({ ...prev, ttft: ((firstTokenTime - startTime) / 1000).toFixed(3) }))
          }
          
          const chunk = decoder.decode(value, { stream: true })
          tokenCount += chunk.trim().split(/\s+/).length
          setStreamingText(prev => prev + chunk)
        }
      }
      
      const endTime = performance.now()
      const totalTime = (endTime - startTime) / 1000
      setMetrics(prev => ({ 
        ...prev, 
        time: totalTime.toFixed(2),
        tps: (tokenCount / totalTime).toFixed(1)
      }))

      // Update telemetry immediately after run
      fetchTelemetry()
    } catch (error) {
      console.error("Error generating text:", error)
      setStreamingText("Error connecting to Inference Lab server on port 8002.")
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="dashboard">
      <header className="header">
        <h1>Inference Lab <span className="badge">v2.0 PRO</span></h1>
        <p className="subtitle">High-Performance LLM Engine Visualization</p>
      </header>
      
      <main className="main-content">
        <div className="control-panel panel">
          <h2><span className="icon">⚙️</span> Engine Settings</h2>
          
          <div className="input-group">
            <label>Inference Engine</label>
            <select value={mode} onChange={e => setMode(e.target.value)}>
              <option value="real">Qwen2.5-0.5B-Instruct (Real PyTorch LLM)</option>
              <option value="batching">Continuous Batching Scheduler (M2)</option>
              <option value="speculative">Speculative Decoding (M3)</option>
              <option value="eviction">Paged Attention / Cache Eviction (M4)</option>
            </select>
          </div>
          
          <div className="input-group">
            <label>Prompt</label>
            <textarea 
              value={prompt} 
              onChange={e => setPrompt(e.target.value)}
              placeholder="Enter your system prompt or instruction..."
              rows={4}
            />
          </div>

          <div className="sliders-grid">
            <div className="input-group">
              <label>Max Tokens: <span className="val-display">{maxTokens}</span></label>
              <input 
                type="range" min="10" max="1024" value={maxTokens} 
                onChange={e => setMaxTokens(e.target.value)}
              />
            </div>
            
            <div className="input-group">
              <label>Temperature: <span className="val-display">{temperature}</span></label>
              <input 
                type="range" min="0" max="2" step="0.1" value={temperature} 
                onChange={e => setTemperature(e.target.value)}
              />
            </div>

            <div className="input-group">
              <label>Top P: <span className="val-display">{topP}</span></label>
              <input 
                type="range" min="0" max="1" step="0.05" value={topP} 
                onChange={e => setTopP(e.target.value)}
              />
            </div>
          </div>

          <button 
            className={`generate-btn ${isGenerating ? 'generating' : ''}`}
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            {isGenerating ? 'RUNNING INFERENCE...' : 'INITIALIZE STREAM'}
          </button>
        </div>

        <div className="metrics-panel">
          <div className="metric-card">
            <div className="metric-label">TTFT (Latency)</div>
            <div className="metric-value">{metrics.ttft ? `${metrics.ttft}s` : '--'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Tokens / Second</div>
            <div className="metric-value">{metrics.tps ? `${metrics.tps}` : '--'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Total Time</div>
            <div className="metric-value">{metrics.time ? `${metrics.time}s` : '--'}</div>
          </div>

          {telemetry && (
            <div className="metric-card" style={{ gridColumn: 'span 3', textAlign: 'left', padding: '1rem 1.5rem' }}>
              <div className="metric-label" style={{ marginBottom: '0.4rem', color: 'var(--accent-primary)' }}>
                ⚡ Engine Telemetry ({mode.toUpperCase()})
              </div>
              <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                {mode === 'speculative' && telemetry.speculative && (
                  <>
                    <span>Acceptance Rate: <strong style={{ color: '#fff' }}>{(telemetry.speculative.acceptance_rate * 100).toFixed(1)}%</strong></span>
                    <span>Tokens/Step: <strong style={{ color: '#fff' }}>{telemetry.speculative.tokens_per_step}x</strong></span>
                    <span>Draft Tokens: <strong style={{ color: '#fff' }}>{telemetry.speculative.total_draft_tokens}</strong></span>
                  </>
                )}
                {mode === 'eviction' && telemetry.cache_eviction && (
                  <>
                    <span>Budget: <strong style={{ color: '#fff' }}>{telemetry.cache_eviction.budget_tokens} tok</strong></span>
                    <span>Sinks Kept: <strong style={{ color: '#fff' }}>{telemetry.cache_eviction.num_sink_tokens}</strong></span>
                    <span>Evicted Tokens: <strong style={{ color: '#fff' }}>{telemetry.cache_eviction.total_tokens_evicted}</strong></span>
                    <span>Peak KV VRAM: <strong style={{ color: '#fff' }}>{telemetry.cache_eviction.peak_vram_mb} MB</strong></span>
                  </>
                )}
                {mode === 'batching' && telemetry.batching && (
                  <>
                    <span>Batched Tokens: <strong style={{ color: '#fff' }}>{telemetry.batching.total_tokens}</strong></span>
                    <span>Batches Processed: <strong style={{ color: '#fff' }}>{telemetry.batching.total_batches}</strong></span>
                  </>
                )}
                {mode === 'real' && (
                  <>
                    <span>Model: <strong style={{ color: '#fff' }}>{telemetry.model}</strong></span>
                    <span>Device: <strong style={{ color: '#fff' }}>{telemetry.device.toUpperCase()}</strong></span>
                    <span>Scheduler: <strong style={{ color: '#fff' }}>PyTorch Autoregressive</strong></span>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="output-panel panel">
          <div className="panel-header-row">
            <h2><span className="icon">📡</span> Output Stream</h2>
            <div className={`status-indicator ${isGenerating ? 'active' : ''}`}>
              <div className="dot"></div>
              {isGenerating ? 'Streaming...' : 'Idle'}
            </div>
          </div>
          <div className="terminal">
            {streamingText || <span className="placeholder-text">System awaiting execution order...</span>}
            {isGenerating && <span className="cursor" />}
            <div ref={endRef} />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
