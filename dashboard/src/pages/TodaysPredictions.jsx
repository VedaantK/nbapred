import { useState, useEffect, useRef } from 'react'
import { RefreshCw, Zap, AlertCircle, CheckCircle, Calendar } from 'lucide-react'
import PredictionCard from '../components/PredictionCard.jsx'
import EdgeTable from '../components/EdgeTable.jsx'

const STEP_LABELS = {
  sportsbook: 'Fetching sportsbook lines...',
  kalshi:     'Fetching Kalshi odds...',
  polymarket: 'Fetching Polymarket odds...',
  predicting: 'Generating predictions...',
  done:       'Done!',
  error:      'Error',
}

function StatCard({ label, value, sub, color = 'text-white' }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  )
}

function todayStr() {
  return new Date().toISOString().split('T')[0]
}

export default function TodaysPredictions() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [pipelineState, setPipelineState] = useState(null)
  const [selectedDate, setSelectedDate] = useState(todayStr)
  const [view, setView] = useState('table')
  const pollRef = useRef(null)

  const running = pipelineState?.status === 'running'

  async function load(dateOverride) {
    const d = dateOverride ?? selectedDate
    setLoading(true)
    try {
      const res = await fetch(`/api/predictions/today?game_date=${d}`)
      if (res.ok) setData(await res.json())
    } catch {}
    setLoading(false)
  }

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  function startPolling() {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch('/api/pipeline/status')
        if (!res.ok) return
        const state = await res.json()
        setPipelineState(state)

        if (state.status === 'done') {
          stopPolling()
          load(state.game_date || selectedDate)
        } else if (state.status === 'error') {
          stopPolling()
        }
      } catch {}
    }, 2000)
  }

  async function runPipeline() {
    setPipelineState({ status: 'running', step: 'sportsbook', message: 'Fetching sportsbook lines...' })
    try {
      const res = await fetch(`/api/pipeline/run?game_date=${selectedDate}`, { method: 'POST' })
      const body = await res.json()
      if (body.status === 'error') {
        setPipelineState({ status: 'error', step: 'error', message: body.message })
        return
      }
    } catch (e) {
      setPipelineState({ status: 'error', step: 'error', message: String(e) })
      return
    }
    startPolling()
  }

  // On mount: check if a pipeline is already running
  useEffect(() => {
    fetch('/api/pipeline/status')
      .then(r => r.json())
      .then(state => {
        if (state.status === 'running') {
          setPipelineState(state)
          startPolling()
        }
      })
      .catch(() => {})
    return stopPolling
  }, [])

  // Reload predictions whenever the selected date changes
  useEffect(() => {
    load(selectedDate)
  }, [selectedDate])

  const predictions = data?.predictions ?? []
  const highConf = predictions.filter(p => p.confidence === 'high')
  const bestEdge = predictions.length > 0 ? predictions[0] : null

  const displayDate = new Date(selectedDate + 'T12:00:00').toLocaleDateString('en-US', {
    weekday: 'long', month: 'long', day: 'numeric', year: 'numeric',
  })

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Predictions</h1>
          <p className="text-sm text-gray-500 mt-0.5">{displayDate}</p>
        </div>
        <div className="flex gap-2 items-center">
          {/* Date picker */}
          <div className="flex items-center gap-1.5 bg-gray-900 border border-gray-700 rounded px-2 py-1.5">
            <Calendar size={14} className="text-gray-400" />
            <input
              type="date"
              value={selectedDate}
              onChange={e => setSelectedDate(e.target.value)}
              className="bg-transparent text-sm text-white outline-none cursor-pointer"
            />
          </div>

          <button
            onClick={runPipeline}
            disabled={running}
            className="flex items-center gap-1.5 text-sm px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded text-white transition-colors"
          >
            <RefreshCw size={14} className={running ? 'animate-spin' : ''} />
            {running ? (STEP_LABELS[pipelineState?.step] ?? 'Running...') : 'Run Pipeline'}
          </button>

          <div className="flex border border-gray-700 rounded overflow-hidden text-sm">
            {['table', 'cards'].map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-3 py-1.5 transition-colors ${view === v ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                {v.charAt(0).toUpperCase() + v.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Pipeline status banner */}
      {pipelineState && pipelineState.status !== 'idle' && (
        <div className={`flex items-start gap-2 rounded-lg px-4 py-3 mb-4 text-sm border ${
          pipelineState.status === 'error'
            ? 'bg-red-950 border-red-800 text-red-300'
            : pipelineState.status === 'done'
            ? 'bg-green-950 border-green-800 text-green-300'
            : 'bg-blue-950 border-blue-800 text-blue-300'
        }`}>
          {pipelineState.status === 'error'   && <AlertCircle size={16} className="mt-0.5 shrink-0" />}
          {pipelineState.status === 'done'    && <CheckCircle size={16} className="mt-0.5 shrink-0" />}
          {pipelineState.status === 'running' && <RefreshCw size={16} className="mt-0.5 shrink-0 animate-spin" />}
          <div>
            {pipelineState.status === 'running' ? (
              <span>{pipelineState.message || (STEP_LABELS[pipelineState.step] ?? 'Running...')}</span>
            ) : (
              <>
                <span className="font-medium">
                  {pipelineState.status === 'error' ? 'Pipeline failed — ' : 'Pipeline complete — '}
                </span>
                {pipelineState.message && (
                  <span className="opacity-80">{pipelineState.message}</span>
                )}
              </>
            )}
          </div>
          {(pipelineState.status === 'done' || pipelineState.status === 'error') && (
            <button
              className="ml-auto opacity-50 hover:opacity-100 text-xs"
              onClick={() => setPipelineState(null)}
            >
              ✕
            </button>
          )}
        </div>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <StatCard label="Total Predictions" value={predictions.length} />
        <StatCard
          label="High Confidence"
          value={highConf.length}
          color="text-green-400"
        />
        <StatCard
          label="Best Edge"
          value={bestEdge ? `${bestEdge.player_name_display || bestEdge.player_name}` : '—'}
          sub={bestEdge ? `${(bestEdge.edge * 100).toFixed(1)}% edge` : ''}
          color="text-blue-400"
        />
        <StatCard
          label="Avg Edge"
          value={predictions.length > 0
            ? `${(predictions.reduce((s, p) => s + Math.abs(p.edge ?? 0), 0) / predictions.length * 100).toFixed(1)}%`
            : '—'}
        />
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading predictions...</div>
        </div>
      ) : predictions.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-12 text-center">
          <Zap size={32} className="text-gray-700 mx-auto mb-3" />
          <p className="text-gray-500">No predictions for {displayDate}.</p>
          <p className="text-gray-600 text-sm mt-1">Click "Run Pipeline" to fetch odds and generate predictions.</p>
        </div>
      ) : view === 'table' ? (
        <EdgeTable predictions={predictions} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {predictions.map((p, i) => (
            <PredictionCard key={i} pred={p} />
          ))}
        </div>
      )}
    </div>
  )
}
