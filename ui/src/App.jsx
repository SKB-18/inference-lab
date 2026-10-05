import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('Explain the concept of continuous batching.')
  const [maxTokens, setMaxTokens] = useState(50)
  const [streamingText, setStreamingText] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [metrics, setMetrics] = useState({ ttft: null, tps: null, time: null })
  
  const endRef = useRef(null)

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
      const response = await fetch('http://127.0.0.1:8000/v1/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt, max_tokens: parseInt(maxTokens) })
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
          // Assuming words separated by spaces as rough token approximation for this dummy UI
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
    } catch (error) {
      console.error("Error generating text:", error)
      setStreamingText("Error connecting to Inference Lab server. Is it running on port 8000?")
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="dashboard">
      <header className="header">
        <h1>Inference Lab <span className="badge">M2 Batching</span></h1>
        <p className="subtitle">Real-time LLM Engine Visualization</p>
      </header>
      
      <main className="main-content">
        <div className="control-panel panel">
          <h2>Request Controls</h2>
          <div className="input-group">
            <label>Prompt</label>
            <textarea 
              value={prompt} 
              onChange={e => setPrompt(e.target.value)}
              placeholder="Enter your prompt..."
              rows={4}
            />
          </div>
          <div className="input-group">
            <label>Max Tokens: {maxTokens}</label>
            <input 
              type="range" 
              min="10" 
              max="500" 
              value={maxTokens} 
              onChange={e => setMaxTokens(e.target.value)}
            />
          </div>
          <button 
            className={`generate-btn ${isGenerating ? 'generating' : ''}`}
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generate Stream'}
          </button>
        </div>

        <div className="metrics-panel">
          <div className="metric-card">
            <div className="metric-label">TTFT (Time To First Token)</div>
            <div className="metric-value">{metrics.ttft ? `${metrics.ttft}s` : '--'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Tokens Per Second</div>
            <div className="metric-value">{metrics.tps ? `${metrics.tps} tok/s` : '--'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Total Time</div>
            <div className="metric-value">{metrics.time ? `${metrics.time}s` : '--'}</div>
          </div>
        </div>

        <div className="output-panel panel">
          <h2>Stream Output</h2>
          <div className="terminal">
            {streamingText || <span className="placeholder-text">Awaiting input...</span>}
            {isGenerating && <span className="cursor" />}
            <div ref={endRef} />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
