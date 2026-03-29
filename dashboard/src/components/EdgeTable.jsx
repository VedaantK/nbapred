import { useState } from 'react'
import { ChevronUp, ChevronDown } from 'lucide-react'

function SortIcon({ col, sortCol, sortDir }) {
  if (sortCol !== col) return <span className="text-gray-700 ml-1">⇅</span>
  return sortDir === 'asc'
    ? <ChevronUp size={13} className="inline ml-1 text-blue-400" />
    : <ChevronDown size={13} className="inline ml-1 text-blue-400" />
}

const CONF_ORDER = { high: 0, medium: 1, low: 2 }

/**
 * Sortable, filterable table of predictions with edges highlighted.
 * predictions: array from /api/predictions/today
 */
export default function EdgeTable({ predictions = [] }) {
  const [sortCol, setSortCol] = useState('edge')
  const [sortDir, setSortDir] = useState('desc')
  const [filter, setFilter] = useState({ confidence: 'all', minEdge: '' })

  function toggleSort(col) {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('desc') }
  }

  const filtered = predictions.filter(p => {
    if (filter.confidence !== 'all' && p.confidence !== filter.confidence) return false
    if (filter.minEdge !== '' && Math.abs(p.edge ?? 0) < parseFloat(filter.minEdge) / 100) return false
    return true
  })

  const sorted = [...filtered].sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol]
    if (sortCol === 'confidence') { va = CONF_ORDER[va] ?? 9; vb = CONF_ORDER[vb] ?? 9 }
    if (sortCol === 'edge') { va = Math.abs(va ?? 0); vb = Math.abs(vb ?? 0) }
    if (va == null) return 1
    if (vb == null) return -1
    return sortDir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1)
  })

  function edgeStyle(edge) {
    const abs = Math.abs(edge ?? 0)
    if (abs >= 0.10) return 'text-green-400 font-bold'
    if (abs >= 0.05) return 'text-yellow-400'
    return 'text-gray-500'
  }

  function confBadge(c) {
    const s = { high: 'bg-green-900 text-green-300', medium: 'bg-yellow-900 text-yellow-300', low: 'bg-gray-800 text-gray-400' }
    return <span className={`text-xs px-1.5 py-0.5 rounded ${s[c] || s.low}`}>{c?.toUpperCase()}</span>
  }

  const TH = ({ col, children }) => (
    <th
      className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer select-none hover:text-gray-300"
      onClick={() => toggleSort(col)}
    >
      {children}<SortIcon col={col} sortCol={sortCol} sortDir={sortDir} />
    </th>
  )

  return (
    <div>
      {/* Filters */}
      <div className="flex gap-3 mb-3">
        <select
          className="bg-gray-800 border border-gray-700 text-sm text-gray-300 rounded px-2 py-1"
          value={filter.confidence}
          onChange={e => setFilter(f => ({ ...f, confidence: e.target.value }))}
        >
          <option value="all">All Confidence</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
        <input
          type="number"
          placeholder="Min edge %"
          className="bg-gray-800 border border-gray-700 text-sm text-gray-300 rounded px-2 py-1 w-28"
          value={filter.minEdge}
          onChange={e => setFilter(f => ({ ...f, minEdge: e.target.value }))}
        />
        <span className="text-xs text-gray-500 self-center">{sorted.length} results</span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-800">
        <table className="w-full text-sm">
          <thead className="bg-gray-900">
            <tr>
              <TH col="player_name">Player</TH>
              <TH col="predicted_points">Predicted</TH>
              <TH col="sportsbook_line">Line</TH>
              <TH col="predicted_over_prob">Model %</TH>
              <TH col="sportsbook_implied_prob">Book %</TH>
              <TH col="edge">Edge</TH>
              <TH col="confidence">Confidence</TH>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800 bg-gray-950">
            {sorted.length === 0 && (
              <tr>
                <td colSpan={7} className="px-3 py-8 text-center text-gray-600">
                  No predictions match your filters
                </td>
              </tr>
            )}
            {sorted.map((p, i) => (
              <tr key={i} className="hover:bg-gray-900 transition-colors">
                <td className="px-3 py-2.5 font-medium text-white">
                  {p.player_name_display || p.player_name}
                </td>
                <td className="px-3 py-2.5 text-gray-300">{Math.round(p.predicted_points)} pts</td>
                <td className="px-3 py-2.5 text-gray-400">{p.sportsbook_line}</td>
                <td className="px-3 py-2.5 text-blue-400">{p.predicted_over_prob != null ? `${Math.round(p.predicted_over_prob * 100)}%` : '—'}</td>
                <td className="px-3 py-2.5 text-gray-400">{p.sportsbook_implied_prob != null ? `${Math.round(p.sportsbook_implied_prob * 100)}%` : '—'}</td>
                <td className={`px-3 py-2.5 ${edgeStyle(p.edge)}`}>
                  {p.edge != null ? `${p.edge >= 0 ? '+' : ''}${(p.edge * 100).toFixed(1)}%` : '—'}
                </td>
                <td className="px-3 py-2.5">{confBadge(p.confidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
