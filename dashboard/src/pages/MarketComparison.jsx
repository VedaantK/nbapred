import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine, Cell,
} from 'recharts'

function AccBar({ label, accuracy, color }) {
  if (accuracy == null) return null
  const pct = Math.round(accuracy * 100)
  return (
    <div className="flex items-center gap-3">
      <span className="text-sm text-gray-400 w-32 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-800 rounded h-3 overflow-hidden">
        <div className="h-full rounded transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-sm font-bold w-10 text-right" style={{ color }}>{pct}%</span>
    </div>
  )
}

export default function MarketComparison() {
  const [data, setData] = useState(null)
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(true)

  async function load() {
    setLoading(true)
    try {
      const res = await fetch(`/api/performance/comparison?days=${days}`)
      if (res.ok) setData(await res.json())
    } catch {}
    setLoading(false)
  }

  useEffect(() => { load() }, [days])

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <span className="text-gray-500">Loading comparison data...</span>
    </div>
  )

  if (data?.error) return (
    <div className="flex items-center justify-center h-64">
      <div className="text-center">
        <p className="text-gray-500">{data.error}</p>
        <p className="text-gray-600 text-sm mt-1">Predictions need to be scored against actual results first.</p>
      </div>
    </div>
  )

  const barData = [
    { name: 'Your Model', accuracy: data?.model_accuracy, fill: '#3b82f6' },
    { name: 'Sportsbook', accuracy: data?.sportsbook_accuracy, fill: '#6b7280' },
    { name: 'Kalshi', accuracy: data?.kalshi_accuracy, fill: '#a855f7' },
    { name: 'Polymarket', accuracy: data?.polymarket_accuracy, fill: '#ec4899' },
  ].filter(d => d.accuracy != null).map(d => ({ ...d, pct: Math.round(d.accuracy * 100) }))

  const edgeAnalysis = Object.entries(data?.edge_analysis || {}).map(([key, val]) => ({
    label: key.replace('edge_', '').replace('pct', '%'),
    ...val,
    accuracy_pct: Math.round((val.model_accuracy || 0) * 100),
  }))

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Market Comparison</h1>
          <p className="text-sm text-gray-500 mt-0.5">Your model vs sportsbooks vs prediction markets</p>
        </div>
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

      <div className="grid md:grid-cols-2 gap-4 mb-4">
        {/* Accuracy bar chart */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Accuracy Comparison</h3>
          {barData.length === 0 ? (
            <div className="h-48 flex items-center justify-center text-gray-600">No data yet</div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} />
                <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#6b7280', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                  formatter={v => [`${v}%`]}
                />
                <ReferenceLine y={50} stroke="#374151" strokeDasharray="3 3" />
                <Bar dataKey="pct" radius={[4, 4, 0, 0]}>
                  {barData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Accuracy detail list */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Detailed Accuracy</h3>
          <div className="space-y-3">
            <AccBar label="Your Model" accuracy={data?.model_accuracy} color="#3b82f6" />
            <AccBar label="Sportsbook" accuracy={data?.sportsbook_accuracy} color="#6b7280" />
            <AccBar label="Kalshi" accuracy={data?.kalshi_accuracy} color="#a855f7" />
            <AccBar label="Polymarket" accuracy={data?.polymarket_accuracy} color="#ec4899" />
          </div>
          {data?.total_predictions != null && (
            <p className="text-xs text-gray-600 mt-4">{data.total_predictions} predictions over {days} days</p>
          )}
        </div>
      </div>

      {/* Edge analysis */}
      {edgeAnalysis.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-4">
            Edge Analysis — When Model Found an Edge, What Was the Hit Rate?
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  {['Min Edge', 'Bets', 'Hit Rate', 'vs. Baseline'].map(h => (
                    <th key={h} className="px-3 py-2 text-left text-xs text-gray-500 uppercase">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {edgeAnalysis.map((row, i) => (
                  <tr key={i}>
                    <td className="px-3 py-2 text-gray-300 font-mono">{row.label}</td>
                    <td className="px-3 py-2 text-gray-400">{row.count}</td>
                    <td className={`px-3 py-2 font-bold ${row.accuracy_pct > 55 ? 'text-green-400' : 'text-gray-300'}`}>
                      {row.accuracy_pct}%
                    </td>
                    <td className={`px-3 py-2 text-sm ${row.accuracy_pct > 52.4 ? 'text-green-400' : 'text-red-400'}`}>
                      {row.accuracy_pct > 52.4 ? '▲' : '▼'} {Math.abs(row.accuracy_pct - 52).toFixed(1)}%
                      <span className="text-gray-600 ml-1">(vs 52%)</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
