import { useState, useEffect, useRef } from 'react'
import { Calendar, RefreshCw, CheckCircle2, XCircle, TrendingUp, Target, Cpu } from 'lucide-react'

// ── helpers ──────────────────────────────────────────────────────────────────

function yesterday() {
  const d = new Date()
  d.setDate(d.getDate() - 1)
  return d.toISOString().slice(0, 10)
}

function errorColor(absError) {
  if (absError <= 3) return 'text-green-400'
  if (absError <= 6) return 'text-yellow-400'
  return 'text-red-400'
}

// ── sub-components ────────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, color = 'text-white' }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>{value ?? '—'}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  )
}

function ResultBadge({ went_over, model_predicted_over, actual_points }) {
  if (actual_points == null) return <span className="text-gray-600 text-xs">Pending</span>
  const direction = went_over ? 'OVER' : 'UNDER'
  const correct = went_over === model_predicted_over
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded ${
      correct ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
    }`}>
      {correct ? <CheckCircle2 size={11} /> : <XCircle size={11} />}
      {correct ? '✓' : '✗'} {direction}
    </span>
  )
}

function ConfBadge({ confidence }) {
  const colors = {
    high: 'bg-blue-900 text-blue-300',
    medium: 'bg-gray-800 text-gray-400',
    low: 'bg-gray-900 text-gray-600',
  }
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded capitalize ${colors[confidence] || colors.low}`}>
      {confidence}
    </span>
  )
}

function ResultsTable({ players }) {
  if (!players || players.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 text-center text-gray-500 text-sm">
        No results yet for this date — click "Pull Results" to fetch from the NBA API
      </div>
    )
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 text-xs text-gray-500">
            <th className="text-left px-4 py-2.5 font-medium">Player</th>
            <th className="text-right px-3 py-2.5 font-medium">Line</th>
            <th className="text-right px-3 py-2.5 font-medium">Predicted</th>
            <th className="text-right px-3 py-2.5 font-medium">Actual</th>
            <th className="text-right px-3 py-2.5 font-medium">Error</th>
            <th className="text-center px-3 py-2.5 font-medium">Result</th>
            <th className="text-center px-3 py-2.5 font-medium">Confidence</th>
            <th className="text-right px-3 py-2.5 font-medium">Edge</th>
          </tr>
        </thead>
        <tbody>
          {players.map((p, i) => {
            const absErr = p.actual_points != null
              ? Math.abs((p.actual_points ?? 0) - (p.predicted_points ?? 0))
              : null
            return (
              <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                <td className="px-4 py-2.5 text-white font-medium">{p.player_name}</td>
                <td className="px-3 py-2.5 text-right text-gray-400">
                  {p.sportsbook_line ?? '—'}
                </td>
                <td className="px-3 py-2.5 text-right text-blue-300">
                  {p.predicted_points != null ? Math.round(p.predicted_points) : '—'}
                </td>
                <td className="px-3 py-2.5 text-right text-white font-medium">
                  {p.actual_points ?? '—'}
                </td>
                <td className={`px-3 py-2.5 text-right font-medium ${absErr != null ? errorColor(absErr) : 'text-gray-600'}`}>
                  {absErr != null
                    ? (p.actual_points - p.predicted_points > 0 ? '+' : '') +
                      (p.actual_points - Math.round(p.predicted_points))
                    : '—'}
                </td>
                <td className="px-3 py-2.5 text-center">
                  <ResultBadge
                    went_over={p.went_over}
                    model_predicted_over={p.model_predicted_over}
                    actual_points={p.actual_points}
                  />
                </td>
                <td className="px-3 py-2.5 text-center">
                  <ConfBadge confidence={p.confidence} />
                </td>
                <td className={`px-3 py-2.5 text-right font-medium ${
                  p.edge > 0.08 ? 'text-green-400' : p.edge < -0.08 ? 'text-red-400' : 'text-gray-400'
                }`}>
                  {p.edge != null ? (p.edge > 0 ? '+' : '') + (p.edge * 100).toFixed(1) + '%' : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── main component ────────────────────────────────────────────────────────────

function useCountdown(isoTarget) {
  const [secs, setSecs] = useState(null)
  useEffect(() => {
    if (!isoTarget) { setSecs(null); return }
    const tick = () => {
      const diff = Math.max(0, Math.floor((new Date(isoTarget) - Date.now()) / 1000))
      setSecs(diff)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [isoTarget])
  if (secs == null) return null
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

export default function DailyResults() {
  const [selectedDate, setSelectedDate] = useState(yesterday())
  const [data, setData] = useState(null)
  const [fetching, setFetching] = useState(false)
  const [loading, setLoading] = useState(false)
  const [fetchState, setFetchState] = useState({ status: 'idle', message: '', attempt: 0, max_attempts: 5, scored: 0, next_retry: null })
  const [scoringState, setScoringState] = useState({ status: 'idle', message: '' })
  const [retrainState, setRetrainState] = useState({ status: 'idle', message: '' })
  const fetchPollRef = useRef(null)
  const scoringPollRef = useRef(null)
  const retrainPollRef = useRef(null)

  const countdown = useCountdown(fetchState.next_retry)

  async function loadResults(date) {
    setLoading(true)
    try {
      const res = await fetch(`/api/results/daily?game_date=${date}`)
      if (res.ok) setData(await res.json())
    } catch {}
    setLoading(false)
  }

  async function pollFetchStatus(date) {
    try {
      const res = await fetch('/api/results/fetch/status')
      if (!res.ok) return
      const state = await res.json()
      setFetchState(state)
      if (state.status === 'done') {
        clearInterval(fetchPollRef.current)
        fetchPollRef.current = null
        setFetching(false)
        await loadResults(date)
      } else if (state.status === 'error') {
        clearInterval(fetchPollRef.current)
        fetchPollRef.current = null
        setFetching(false)
      }
    } catch {}
  }

  async function pullResults() {
    setFetching(true)
    setFetchState({ status: 'running', message: 'Starting...', attempt: 0, max_attempts: 5, scored: 0, next_retry: null })
    try {
      await fetch(`/api/results/fetch?game_date=${selectedDate}`, { method: 'POST' })
    } catch {}
    fetchPollRef.current = setInterval(() => pollFetchStatus(selectedDate), 5000)
  }

  async function fetchScoringStatus() {
    try {
      const res = await fetch('/api/scoring/status')
      if (!res.ok) return
      const d = await res.json()
      setScoringState(d)
      if (d.status === 'done' || d.status === 'error') {
        clearInterval(scoringPollRef.current)
        scoringPollRef.current = null
        if (d.status === 'done') loadResults(selectedDate)
      }
    } catch {}
  }

  async function fetchRetrainStatus() {
    try {
      const res = await fetch('/api/training/status')
      if (!res.ok) return
      const d = await res.json()
      setRetrainState(d)
      if (d.status === 'done' || d.status === 'error') {
        clearInterval(retrainPollRef.current)
        retrainPollRef.current = null
      }
    } catch {}
  }

  async function runDailyLearning() {
    setScoringState({ status: 'running', message: `Scoring predictions for ${selectedDate}...` })
    try {
      await fetch(`/api/scoring/run?game_date=${selectedDate}`, { method: 'POST' })
    } catch {}
    scoringPollRef.current = setInterval(fetchScoringStatus, 2000)
  }

  async function runFullRetrain() {
    setRetrainState({ status: 'running', message: 'Retraining all 4 models from scratch (this takes 5-30 min)...' })
    try {
      await fetch('/api/training/run', { method: 'POST' })
    } catch {}
    retrainPollRef.current = setInterval(fetchRetrainStatus, 3000)
  }

  useEffect(() => {
    loadResults(selectedDate)
    if (scoringPollRef.current) {
      clearInterval(scoringPollRef.current)
      scoringPollRef.current = null
    }
    setScoringState({ status: 'idle', message: '' })
  }, [selectedDate])

  useEffect(() => {
    return () => {
      if (fetchPollRef.current) clearInterval(fetchPollRef.current)
      if (scoringPollRef.current) clearInterval(scoringPollRef.current)
      if (retrainPollRef.current) clearInterval(retrainPollRef.current)
    }
  }, [])

  const summary = data?.summary || {}
  const players = data?.players || []
  const fetched = data?.fetched ?? false

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-white">Daily Results</h1>
        <div className="flex items-center gap-2">
          <Calendar size={14} className="text-gray-500" />
          <input
            type="date"
            value={selectedDate}
            onChange={e => setSelectedDate(e.target.value)}
            className="bg-gray-800 border border-gray-700 text-sm text-gray-300 rounded px-2 py-1.5"
          />
          <button
            onClick={pullResults}
            disabled={fetching}
            className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
          >
            <RefreshCw size={13} className={fetching ? 'animate-spin' : ''} />
            {fetching ? 'Fetching...' : fetched ? 'Refresh Results' : 'Pull Results'}
          </button>
        </div>
      </div>

      {/* Fetch status panel — shown while fetching or on result/error */}
      {fetchState.status !== 'idle' && (
        <div className={`rounded-lg border px-4 py-3 mb-5 flex items-start gap-3 text-sm ${
          fetchState.status === 'done'  ? 'bg-green-900/20 border-green-800 text-green-300' :
          fetchState.status === 'error' ? 'bg-red-900/20 border-red-800 text-red-300' :
                                          'bg-blue-900/20 border-blue-800 text-blue-300'
        }`}>
          {fetchState.status === 'running' && <RefreshCw size={14} className="animate-spin mt-0.5 shrink-0" />}
          {fetchState.status === 'done'    && <CheckCircle2 size={14} className="mt-0.5 shrink-0" />}
          {fetchState.status === 'error'   && <XCircle size={14} className="mt-0.5 shrink-0" />}
          <div className="flex-1">
            <div className="font-medium">{fetchState.message}</div>
            {fetchState.status === 'running' && fetchState.next_retry && countdown && (
              <div className="text-xs mt-0.5 opacity-75">
                Next retry in {countdown}
              </div>
            )}
            {fetchState.status === 'running' && fetchState.attempt > 0 && (
              <div className="text-xs mt-0.5 opacity-60">
                The NBA API typically finalises box scores 15–30 min after the last game ends.
              </div>
            )}
          </div>
          {fetchState.status !== 'running' && (
            <button
              onClick={() => setFetchState({ status: 'idle', message: '', attempt: 0, max_attempts: 5, scored: 0, next_retry: null })}
              className="text-xs opacity-50 hover:opacity-100 transition-opacity"
            >✕</button>
          )}
        </div>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="Predictions Scored"
          value={summary.count ?? (loading ? '...' : '—')}
          sub="players with results"
        />
        <MetricCard
          label="Correct (Over/Under)"
          value={summary.count ? `${summary.correct} / ${summary.count}` : '—'}
          sub="directional calls"
          color={summary.directional_accuracy > 0.55 ? 'text-green-400' : summary.directional_accuracy > 0.45 ? 'text-yellow-400' : 'text-red-400'}
        />
        <MetricCard
          label="Directional Accuracy"
          value={summary.directional_accuracy != null ? `${(summary.directional_accuracy * 100).toFixed(1)}%` : '—'}
          sub="over/under correct rate"
          color={summary.directional_accuracy > 0.55 ? 'text-green-400' : 'text-white'}
        />
        <MetricCard
          label="Mean Absolute Error"
          value={summary.mae != null ? `${summary.mae} pts` : '—'}
          sub="avg pts off per player"
          color={summary.mae < 4 ? 'text-green-400' : summary.mae < 6 ? 'text-yellow-400' : 'text-red-400'}
        />
      </div>

      {/* Results table */}
      <div className="mb-6">
        <h2 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wide">
          Player Results — {selectedDate}
        </h2>
        {loading
          ? <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 text-center text-gray-500 text-sm">Loading...</div>
          : <ResultsTable players={players} />
        }
      </div>

      {/* Daily Learning section */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-4">
        <div className="flex items-center gap-2 mb-1">
          <TrendingUp size={15} className="text-blue-400" />
          <h2 className="text-sm font-semibold text-gray-300">Daily Learning</h2>
        </div>

        {/* Fast: score + update weights */}
        <div>
          <p className="text-xs text-gray-500 mb-2">
            <span className="text-gray-300 font-medium">Score &amp; Update Weights</span>
            {' '}— Marks yesterday's predictions correct/incorrect and recalculates ensemble weights. Fast (~15s).
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={runDailyLearning}
              disabled={scoringState.status === 'running'}
              className="flex items-center gap-1.5 bg-green-700 hover:bg-green-600 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium px-4 py-2 rounded transition-colors"
            >
              <Target size={13} className={scoringState.status === 'running' ? 'animate-pulse' : ''} />
              {scoringState.status === 'running' ? 'Running...' : 'Run Daily Learning'}
            </button>
            {scoringState.status !== 'idle' && (
              <div className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded ${
                scoringState.status === 'running' ? 'bg-blue-900/50 text-blue-300' :
                scoringState.status === 'done' ? 'bg-green-900/50 text-green-300' :
                'bg-red-900/50 text-red-300'
              }`}>
                {scoringState.status === 'running' && <RefreshCw size={11} className="animate-spin" />}
                {scoringState.status === 'done' && <CheckCircle2 size={11} />}
                {scoringState.status === 'error' && <XCircle size={11} />}
                {scoringState.message || scoringState.status}
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-gray-800 pt-4">
          <p className="text-xs text-gray-500 mb-2">
            <span className="text-gray-300 font-medium">Full Model Retrain</span>
            {' '}— Rebuilds all features and retrains all 4 models from scratch. Run weekly. Takes 5-30 min.
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={runFullRetrain}
              disabled={retrainState.status === 'running'}
              className="flex items-center gap-1.5 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-gray-200 text-sm font-medium px-4 py-2 rounded transition-colors"
            >
              <Cpu size={13} className={retrainState.status === 'running' ? 'animate-pulse' : ''} />
              {retrainState.status === 'running' ? 'Retraining...' : 'Full Retrain'}
            </button>
            {retrainState.status !== 'idle' && (
              <div className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded ${
                retrainState.status === 'running' ? 'bg-blue-900/50 text-blue-300' :
                retrainState.status === 'done' ? 'bg-green-900/50 text-green-300' :
                'bg-red-900/50 text-red-300'
              }`}>
                {retrainState.status === 'running' && <RefreshCw size={11} className="animate-spin" />}
                {retrainState.status === 'done' && <CheckCircle2 size={11} />}
                {retrainState.status === 'error' && <XCircle size={11} />}
                {retrainState.message || retrainState.status}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
