import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { DollarSign, TrendingUp, TrendingDown, Percent } from 'lucide-react'

function MetricCard({ label, value, sub, color = 'text-white' }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const styles = {
    pending: 'bg-gray-800 text-gray-400',
    won: 'bg-green-900 text-green-300',
    lost: 'bg-red-900 text-red-400',
    void: 'bg-gray-800 text-gray-500',
  }
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${styles[status] || styles.pending}`}>
      {status?.toUpperCase()}
    </span>
  )
}

function SourceBadge({ source }) {
  const styles = {
    sportsbook: 'bg-blue-900 text-blue-300',
    kalshi: 'bg-purple-900 text-purple-300',
    polymarket: 'bg-pink-900 text-pink-300',
  }
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${styles[source] || 'bg-gray-800 text-gray-400'}`}>
      {source}
    </span>
  )
}

export default function PaperTrader() {
  const today = new Date().toISOString().split('T')[0]
  const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0]

  const [perf, setPerf] = useState(null)
  const [bets, setBets] = useState([])
  const [betsDate, setBetsDate] = useState(today)
  const [loading, setLoading] = useState(true)
  const [actionStatus, setActionStatus] = useState('')
  const [bankrollInput, setBankrollInput] = useState('30')
  const [days, setDays] = useState(30)

  async function loadPerf() {
    try {
      const res = await fetch(`/api/trader/performance?days=${days}`)
      if (res.ok) setPerf(await res.json())
    } catch {}
  }

  async function loadBets(d) {
    try {
      const res = await fetch(`/api/trader/bets?game_date=${d}`)
      if (res.ok) { const data = await res.json(); setBets(data.bets || []) }
    } catch {}
  }

  async function load() {
    setLoading(true)
    await Promise.all([loadPerf(), loadBets(betsDate)])
    setLoading(false)
  }

  useEffect(() => { load() }, [days])
  useEffect(() => { loadBets(betsDate) }, [betsDate])

  async function handleResetBankroll() {
    const amount = parseFloat(bankrollInput)
    if (isNaN(amount) || amount <= 0) return
    setActionStatus('Resetting...')
    try {
      const res = await fetch(`/api/trader/bankroll/reset?amount=${amount}`, { method: 'POST' })
      if (res.ok) {
        setActionStatus(`Bankroll reset to $${amount.toFixed(2)}`)
        await loadPerf()
      } else {
        setActionStatus('Reset failed')
      }
    } catch { setActionStatus('Reset failed') }
    setTimeout(() => setActionStatus(''), 3000)
  }

  async function handleGenerateBets() {
    setActionStatus(`Generating bets for ${betsDate}...`)
    try {
      const res = await fetch(`/api/trader/bets/generate?game_date=${betsDate}`, { method: 'POST' })
      if (res.ok) {
        const data = await res.json()
        setActionStatus(`Generated ${data.generated} bet${data.generated !== 1 ? 's' : ''}`)
        await Promise.all([loadBets(betsDate), loadPerf()])
      } else {
        const err = await res.json()
        setActionStatus(err.detail || 'Failed to generate bets')
      }
    } catch { setActionStatus('Error generating bets') }
    setTimeout(() => setActionStatus(''), 4000)
  }

  async function handleSettleBets() {
    setActionStatus(`Settling bets for ${betsDate}...`)
    try {
      const res = await fetch(`/api/trader/bets/settle?game_date=${betsDate}`, { method: 'POST' })
      if (res.ok) {
        const data = await res.json()
        setActionStatus(`Settled ${data.settled}: ${data.won}W/${data.lost}L, P&L ${data.net_pnl >= 0 ? '+' : ''}$${data.net_pnl?.toFixed(2)}`)
        await Promise.all([loadBets(betsDate), loadPerf()])
      } else {
        setActionStatus('Settlement failed')
      }
    } catch { setActionStatus('Error settling bets') }
    setTimeout(() => setActionStatus(''), 5000)
  }

  const totalPnl = perf?.total_pnl ?? null
  const pnlColor = totalPnl == null ? 'text-gray-300' : totalPnl > 0 ? 'text-green-400' : totalPnl < 0 ? 'text-red-400' : 'text-gray-300'

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Paper Trader</h1>
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

      {/* KPI cards */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        <MetricCard
          label="Current Bankroll"
          value={perf?.current_balance != null ? `$${perf.current_balance.toFixed(2)}` : '—'}
          sub={perf?.starting_balance != null ? `Started at $${perf.starting_balance.toFixed(2)}` : undefined}
          color="text-white"
        />
        <MetricCard
          label="Total P&L"
          value={totalPnl != null ? `${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}` : '—'}
          sub={perf?.starting_balance != null && totalPnl != null
            ? `${((totalPnl / perf.starting_balance) * 100).toFixed(1)}% return`
            : undefined}
          color={pnlColor}
        />
        <MetricCard
          label="Win Rate"
          value={perf?.win_rate != null ? `${Math.round(perf.win_rate * 100)}%` : '—'}
          sub={perf?.total_bets ? `${perf.total_won}W / ${perf.total_lost}L (${perf.total_bets} total)` : undefined}
          color={perf?.win_rate > 0.5 ? 'text-green-400' : 'text-gray-300'}
        />
      </div>

      {/* Settings + Actions row */}
      <div className="grid md:grid-cols-2 gap-4 mb-6">
        {/* Bankroll settings */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Bankroll Settings</h3>
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">$</span>
            <input
              type="number"
              min="1"
              step="1"
              value={bankrollInput}
              onChange={e => setBankrollInput(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-sm text-gray-300 rounded px-2 py-1.5 w-24"
            />
            <button
              onClick={handleResetBankroll}
              className="bg-gray-700 hover:bg-gray-600 text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
            >
              Reset Bankroll
            </button>
          </div>
          <p className="text-xs text-gray-600 mt-2">Resets bankroll to the specified amount. Existing bets are kept.</p>
        </div>

        {/* Bet actions */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Bet Actions</h3>
          <div className="flex items-center gap-2 flex-wrap">
            <input
              type="date"
              value={betsDate}
              onChange={e => setBetsDate(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-sm text-gray-300 rounded px-2 py-1.5"
            />
            <button
              onClick={handleGenerateBets}
              className="bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
            >
              Generate Bets
            </button>
            <button
              onClick={handleSettleBets}
              className="bg-green-700 hover:bg-green-600 text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
            >
              Settle Bets
            </button>
          </div>
          {actionStatus && (
            <div className="mt-2 text-xs text-blue-300 bg-blue-900/30 rounded px-2 py-1">{actionStatus}</div>
          )}
          <p className="text-xs text-gray-600 mt-2">Generate uses today's predictions. Settle resolves pending bets against actual results.</p>
        </div>
      </div>

      {/* Bankroll history chart */}
      {perf?.daily_balance?.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-4">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Bankroll History</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={perf.daily_balance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="date" tick={{ fill: '#6b7280', fontSize: 10 }} />
              <YAxis tickFormatter={v => `$${v}`} tick={{ fill: '#6b7280', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                formatter={v => [`$${v.toFixed(2)}`, 'Balance']}
              />
              {perf.starting_balance && (
                <ReferenceLine y={perf.starting_balance} stroke="#374151" strokeDasharray="3 3"
                  label={{ value: 'Start', fill: '#4b5563', fontSize: 10 }} />
              )}
              <Line type="monotone" dataKey="balance" stroke="#3b82f6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Performance by source */}
      {perf?.by_source && Object.keys(perf.by_source).length > 0 && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          {['sportsbook', 'kalshi', 'polymarket'].map(src => {
            const s = perf.by_source[src]
            if (!s) return (
              <div key={src} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="text-xs text-gray-500 mb-2 capitalize">{src}</div>
                <div className="text-gray-600 text-sm">No bets yet</div>
              </div>
            )
            return (
              <div key={src} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div className="text-xs text-gray-500 mb-2 capitalize">{src}</div>
                <div className="text-lg font-bold text-white">{s.bets} bets</div>
                <div className="text-xs text-gray-400 mt-1">{s.won}W / {s.lost}L</div>
                <div className={`text-sm font-medium mt-1 ${s.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {s.pnl >= 0 ? '+' : ''}${s.pnl?.toFixed(2)} P&L
                </div>
                {s.roi != null && (
                  <div className="text-xs text-gray-500 mt-0.5">ROI: {(s.roi * 100).toFixed(1)}%</div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Bets table */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg">
        <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-300">
            Bets — {betsDate}
          </h3>
          <span className="text-xs text-gray-500">{bets.length} bets</span>
        </div>
        <div className="overflow-x-auto max-h-96 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-900 sticky top-0">
              <tr>
                {['Player', 'Direction', 'Amount', 'Source', 'Odds', 'Edge', 'Kelly', 'Status', 'P&L'].map(h => (
                  <th key={h} className="px-3 py-2 text-left text-xs text-gray-500 uppercase">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {bets.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-3 py-8 text-center text-gray-600">
                    No bets for this date. Run the pipeline first, then click "Generate Bets".
                  </td>
                </tr>
              )}
              {bets.map((b, i) => (
                <tr key={i} className="hover:bg-gray-800 transition-colors">
                  <td className="px-3 py-2.5 font-medium text-white">{b.player_name}</td>
                  <td className="px-3 py-2.5">
                    <span className={`text-xs font-medium uppercase ${b.bet_direction === 'over' ? 'text-green-400' : 'text-red-400'}`}>
                      {b.bet_direction}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-gray-300">${b.bet_amount?.toFixed(2)}</td>
                  <td className="px-3 py-2.5"><SourceBadge source={b.odds_source} /></td>
                  <td className="px-3 py-2.5 text-gray-400">{b.decimal_odds?.toFixed(3)}</td>
                  <td className="px-3 py-2.5 text-blue-400">{b.edge != null ? `${(b.edge * 100).toFixed(1)}%` : '—'}</td>
                  <td className="px-3 py-2.5 text-gray-500">{b.kelly_fraction != null ? `${(b.kelly_fraction * 100).toFixed(1)}%` : '—'}</td>
                  <td className="px-3 py-2.5"><StatusBadge status={b.status} /></td>
                  <td className="px-3 py-2.5">
                    {b.profit_loss != null ? (
                      <span className={b.profit_loss >= 0 ? 'text-green-400' : 'text-red-400'}>
                        {b.profit_loss >= 0 ? '+' : ''}${b.profit_loss?.toFixed(2)}
                      </span>
                    ) : <span className="text-gray-600">—</span>}
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
