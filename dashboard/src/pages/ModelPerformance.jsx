import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { Calendar, RefreshCw } from 'lucide-react'

function MetricCard({ label, value, sub, color = 'text-white' }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  )
}

function CalibrationChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 h-64 flex items-center justify-center">
        <p className="text-gray-600">Calibration data will appear after predictions are scored</p>
      </div>
    )
  }

  const withRef = data.map(d => ({ ...d, perfect: d.prob_bucket }))

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-4">Model Calibration</h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={withRef}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="prob_bucket" tickFormatter={v => `${Math.round(v * 100)}%`} tick={{ fill: '#6b7280', fontSize: 11 }} />
          <YAxis domain={[0, 1]} tickFormatter={v => `${Math.round(v * 100)}%`} tick={{ fill: '#6b7280', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
            formatter={(v, name) => [`${Math.round(v * 100)}%`, name]}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey="actual_rate" stroke="#3b82f6" name="Actual Hit Rate" dot={{ r: 4 }} strokeWidth={2} />
          <Line type="monotone" dataKey="perfect" stroke="#374151" name="Perfect Calibration" strokeDasharray="4 4" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function ModelBarChart({ data }) {
  if (!data || data.length === 0) return null
  const formatted = data.map(d => ({
    name: d.model_type?.replace('_', ' '),
    accuracy: Math.round((d.accuracy || 0) * 100),
    count: d.count,
  }))
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-4">Accuracy by Model</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={formatted}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} />
          <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#6b7280', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
            formatter={v => [`${v}%`]}
          />
          <ReferenceLine y={50} stroke="#374151" strokeDasharray="3 3" />
          <Bar dataKey="accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function LearningHistoryChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 h-48 flex items-center justify-center">
        <p className="text-gray-600">Retrain history will appear after running Daily Learning</p>
      </div>
    )
  }

  // Build one data point per retrain_date with directional_accuracy per model
  const byDate = {}
  for (const r of data) {
    if (!byDate[r.retrain_date]) byDate[r.retrain_date] = { date: r.retrain_date }
    const modelKey = r.model_name?.replace('_', ' ')
    byDate[r.retrain_date][modelKey] = r.directional_accuracy != null
      ? Math.round(r.directional_accuracy * 100)
      : null
  }
  const chartData = Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date))
  const models = [...new Set(data.map(r => r.model_name?.replace('_', ' ')))]
  const MODEL_COLORS = ['#3b82f6', '#10b981', '#f59e0b']

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-4">Model Improvement Over Time (Directional Accuracy %)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="date" tick={{ fill: '#6b7280', fontSize: 10 }} />
          <YAxis domain={[40, 80]} tickFormatter={v => `${v}%`} tick={{ fill: '#6b7280', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
            formatter={v => v != null ? [`${v}%`] : ['—']}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine y={50} stroke="#374151" strokeDasharray="3 3" label={{ value: '50%', fill: '#4b5563', fontSize: 10 }} />
          {models.map((m, i) => (
            <Line key={m} type="monotone" dataKey={m} stroke={MODEL_COLORS[i % MODEL_COLORS.length]} dot={{ r: 3 }} strokeWidth={2} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

const MODEL_COLORS_OBJ = {
  linear_regression: '#3b82f6',
  random_forest: '#10b981',
  xgboost: '#f59e0b',
}

export default function ModelPerformance() {
  const [summary, setSummary] = useState(null)
  const [calibration, setCalibration] = useState([])
  const [history, setHistory] = useState([])
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(true)

  // Learning / scoring state
  const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0]
  const [scoringDate, setScoringDate] = useState(yesterday)
  const [scoringState, setScoringState] = useState({ status: 'idle', message: '' })
  const [learningHistory, setLearningHistory] = useState([])
  const [errorAnalysis, setErrorAnalysis] = useState(null)
  const scoringPollRef = useRef(null)

  async function load() {
    setLoading(true)
    try {
      const [sumRes, calRes, histRes] = await Promise.all([
        fetch(`/api/performance/summary?days=${days}`),
        fetch('/api/performance/calibration'),
        fetch(`/api/predictions/history?days=${days}`),
      ])
      if (sumRes.ok) setSummary(await sumRes.json())
      if (calRes.ok) { const d = await calRes.json(); setCalibration(d.buckets || []) }
      if (histRes.ok) { const d = await histRes.json(); setHistory(d.predictions || []) }
    } catch {}
    setLoading(false)
  }

  async function loadLearningData() {
    try {
      const [lhRes, eaRes] = await Promise.all([
        fetch('/api/performance/learning-history'),
        fetch(`/api/performance/error-analysis?game_date=${scoringDate}`),
      ])
      if (lhRes.ok) { const d = await lhRes.json(); setLearningHistory(d.history || []) }
      if (eaRes.ok) { const d = await eaRes.json(); if (!d.error) setErrorAnalysis(d) }
    } catch {}
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
        if (d.status === 'done') {
          loadLearningData()
          load()
        }
      }
    } catch {}
  }

  async function runDailyLearning() {
    setScoringState({ status: 'running', message: `Starting scoring for ${scoringDate}...` })
    try {
      await fetch(`/api/scoring/run?game_date=${scoringDate}`, { method: 'POST' })
    } catch {}
    scoringPollRef.current = setInterval(fetchScoringStatus, 2000)
  }

  useEffect(() => { load() }, [days])
  useEffect(() => { loadLearningData() }, [scoringDate])
  useEffect(() => {
    return () => { if (scoringPollRef.current) clearInterval(scoringPollRef.current) }
  }, [])

  const overall = summary?.overall_accuracy
  const roi = summary?.roi
  const byConf = summary?.by_confidence || []
  const highConfAcc = byConf.find(c => c.confidence === 'high')?.accuracy

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Model Performance</h1>
        <select
          className="bg-gray-800 border border-gray-700 text-sm text-gray-300 rounded px-2 py-1.5"
          value={days}
          onChange={e => setDays(Number(e.target.value))}
        >
          {[7, 14, 30, 60, 90].map(d => (
            <option key={d} value={d}>Last {d} days</option>
          ))}
        </select>
      </div>

      {/* Daily Learning trigger */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
        <h2 className="text-sm font-semibold text-gray-300 mb-3">Daily Learning</h2>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <Calendar size={14} className="text-gray-500" />
            <input
              type="date"
              value={scoringDate}
              onChange={e => setScoringDate(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-sm text-gray-300 rounded px-2 py-1"
            />
          </div>
          <button
            onClick={runDailyLearning}
            disabled={scoringState.status === 'running'}
            className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
          >
            <RefreshCw size={13} className={scoringState.status === 'running' ? 'animate-spin' : ''} />
            {scoringState.status === 'running' ? 'Running...' : 'Run Daily Learning'}
          </button>
          {scoringState.status !== 'idle' && (
            <span className={`text-xs px-2 py-0.5 rounded ${
              scoringState.status === 'running' ? 'bg-blue-900 text-blue-300' :
              scoringState.status === 'done' ? 'bg-green-900 text-green-300' :
              'bg-red-900 text-red-300'
            }`}>
              {scoringState.message || scoringState.status}
            </span>
          )}
        </div>
        <p className="text-xs text-gray-600 mt-2">
          Scores predictions vs actual results, analyzes errors, and retrains all models.
        </p>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <MetricCard
          label="Overall Accuracy"
          value={overall != null ? `${Math.round(overall * 100)}%` : '—'}
          color={overall > 0.55 ? 'text-green-400' : 'text-gray-300'}
        />
        <MetricCard
          label="High-Conf Accuracy"
          value={highConfAcc != null ? `${Math.round(highConfAcc * 100)}%` : '—'}
          color={highConfAcc > 0.6 ? 'text-green-400' : 'text-gray-300'}
        />
        <MetricCard
          label="Hypothetical ROI"
          value={roi != null ? `${(roi * 100).toFixed(1)}%` : '—'}
          color={roi > 0 ? 'text-green-400' : roi < 0 ? 'text-red-400' : 'text-gray-300'}
          sub="$100/bet, high+med confidence"
        />
        <MetricCard
          label="Predictions Scored"
          value={summary?.total_predictions ?? '—'}
          sub={`last ${days} days`}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <CalibrationChart data={calibration} />
        <ModelBarChart data={summary?.by_model} />
      </div>

      {/* Learning history */}
      <div className="mb-4">
        <LearningHistoryChart data={learningHistory} />
      </div>

      {/* Error analysis for selected scoring date */}
      {errorAnalysis && (
        <div className="grid md:grid-cols-2 gap-4 mb-4">
          {/* Worst misses */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3">
              Worst Misses — {errorAnalysis.game_date}
            </h3>
            {errorAnalysis.worst_misses?.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    {['Player', 'Predicted', 'Actual', 'Error'].map(h => (
                      <th key={h} className="text-left text-xs text-gray-500 pb-2">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {errorAnalysis.worst_misses.map((m, i) => (
                    <tr key={i}>
                      <td className="py-1.5 text-white">{m.player_name}</td>
                      <td className="py-1.5 text-gray-300">{m.predicted}</td>
                      <td className="py-1.5 text-gray-300">{m.actual}</td>
                      <td className={`py-1.5 font-medium ${m.error > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {m.error > 0 ? '+' : ''}{m.error}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-gray-600 text-sm">No error data yet</p>
            )}
          </div>

          {/* Error stats by confidence */}
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Error Stats by Confidence</h3>
            {errorAnalysis.error_distribution && (
              <div className="mb-3 p-2 bg-gray-800 rounded text-xs text-gray-400 space-y-0.5">
                <div>Overall MAE: <span className="text-white font-medium">{errorAnalysis.error_distribution.mae}</span></div>
                <div>Mean error: <span className="text-white font-medium">{errorAnalysis.error_distribution.mean > 0 ? '+' : ''}{errorAnalysis.error_distribution.mean}</span></div>
                <div>Predictions scored: <span className="text-white font-medium">{errorAnalysis.error_distribution.count}</span></div>
              </div>
            )}
            {Object.entries(errorAnalysis.by_confidence || {}).map(([conf, stats]) => {
              const badgeColor = conf === 'high' ? 'text-green-300' : conf === 'medium' ? 'text-yellow-300' : 'text-gray-400'
              return (
                <div key={conf} className="flex items-center justify-between py-1.5 border-b border-gray-800 last:border-0 text-sm">
                  <span className={`font-medium uppercase text-xs ${badgeColor}`}>{conf}</span>
                  <span className="text-gray-400">MAE: <span className="text-white">{stats.mae}</span></span>
                  <span className="text-gray-400">Acc: <span className="text-white">{Math.round(stats.accuracy * 100)}%</span></span>
                  <span className="text-gray-500 text-xs">{stats.count} bets</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Recent predictions log */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg">
        <div className="px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-medium text-gray-300">Recent Prediction Log</h3>
        </div>
        <div className="overflow-x-auto max-h-80 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-900 sticky top-0">
              <tr>
                {['Date', 'Player', 'Predicted', 'Line', 'Actual', 'Edge', 'Result'].map(h => (
                  <th key={h} className="px-3 py-2 text-left text-xs text-gray-500 uppercase">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {history.length === 0 && (
                <tr><td colSpan={7} className="px-3 py-6 text-center text-gray-600">No history yet</td></tr>
              )}
              {history.slice(0, 100).map((p, i) => (
                <tr key={i} className="hover:bg-gray-800 transition-colors">
                  <td className="px-3 py-2 text-gray-500">{p.game_date}</td>
                  <td className="px-3 py-2 text-white">{p.player_name}</td>
                  <td className="px-3 py-2 text-gray-300">{Math.round(p.predicted_points)}</td>
                  <td className="px-3 py-2 text-gray-400">{p.sportsbook_line}</td>
                  <td className="px-3 py-2 text-gray-300">{p.actual_points ?? '—'}</td>
                  <td className="px-3 py-2 text-gray-400">{p.edge != null ? `${(p.edge * 100).toFixed(1)}%` : '—'}</td>
                  <td className="px-3 py-2">
                    {p.correct == null ? <span className="text-gray-600">Pending</span>
                      : p.correct ? <span className="text-green-400">✓</span>
                      : <span className="text-red-400">✗</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
