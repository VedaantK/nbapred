import { useState, useEffect, useRef } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { DollarSign, TrendingUp, TrendingDown, Percent, Activity, RefreshCw } from 'lucide-react'

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

function timeAgo(isoString) {
  if (!isoString) return null
  const diff = Math.floor((Date.now() - new Date(isoString)) / 1000)
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return `${Math.floor(diff / 3600)}h ago`
}

function timeUntil(isoString) {
  if (!isoString) return null
  const diff = Math.floor((new Date(isoString) - Date.now()) / 1000)
  if (diff <= 0) return 'now'
  if (diff < 60) return `${diff}s`
  return `${Math.floor(diff / 60)}m`
}

export default function PaperTrader() {
  const today = new Date().toISOString().split('T')[0]

  const [perf, setPerf] = useState(null)
  const [bets, setBets] = useState([])
  const [betsDate, setBetsDate] = useState(today)
  const [loading, setLoading] = useState(true)
  const [actionStatus, setActionStatus] = useState('')
  const [bankrollInput, setBankrollInput] = useState('30')
  const [days, setDays] = useState(30)
  const [scannerState, setScannerState] = useState(null)
  const [scanning, setScanning] = useState(false)
  const scanPollRef = useRef(null)

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

  async function loadScannerStatus() {
    try {
      const res = await fetch('/api/trader/scanner/status')
      if (res.ok) setScannerState(await res.json())
    } catch {}
  }

  async function load() {
    setLoading(true)
    await Promise.all([loadPerf(), loadBets(betsDate), loadScannerStatus()])
    setLoading(false)
  }

  useEffect(() => { load() }, [days])
  useEffect(() => { loadBets(betsDate) }, [betsDate])

  // Poll scanner status every 30s
  useEffect(() => {
    scanPollRef.current = setInterval(loadScannerStatus, 30000)
    return () => clearInterval(scanPollRef.current)
  }, [])

  async function handleResetBankroll() {
    const amount = parseFloat(bankrollInput)
    if (isNaN(amount) || amount <= 0) return
    setActionStatus('Resetting...')
    try {
      const res = await fetch(`/api/trader/bankroll/reset?amount=${amount}`, { method: 'POST' })
      if (res.ok) {
        setActionStatus(`Bankroll set to $${amount.toFixed(2)}`)
        await loadPerf()
      } else {
        setActionStatus('Reset failed')
      }
    } catch { setActionStatus('Reset failed') }
    setTimeout(() => setActionStatus(''), 3000)
  }

  async function handleScanNow() {
    setScanning(true)
    setActionStatus('Scanning Kalshi markets...')
    try {
      const res = await fetch('/api/trader/scanner/scan-now', { method: 'POST' })
      if (res.ok) {
        const data = await res.json()
        const msg = `Scanned ${data.scanned} markets — ${data.new_bets} new bet${data.new_bets !== 1 ? 's' : ''} placed`
        setActionStatus(msg)
        await Promise.all([loadBets(betsDate), loadPerf(), loadScannerStatus()])
      } else {
        setActionStatus('Scan failed')
      }
    } catch { setActionStatus('Scan error') }
    setScanning(false)
    setTimeout(() => setActionStatus(''), 5000)
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
  const opportunities = scannerState?.last_opportunities || []

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

      {/* Scanner Status */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Activity size={15} className="text-purple-400" />
            <h3 className="text-sm font-semibold text-gray-300">Kalshi Market Scanner</h3>
            <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${scannerState?.running ? 'bg-green-900 text-green-300' : 'bg-gray-800 text-gray-500'}`}>
              {scannerState?.running ? 'LIVE' : 'IDLE'}
            </span>
          </div>
          <button
            onClick={handleScanNow}
            disabled={scanning}
            className="flex items-center gap-1.5 bg-purple-700 hover:bg-purple-600 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium px-3 py-1.5 rounded transition-colors"
          >
            <RefreshCw size={13} className={scanning ? 'animate-spin' : ''} />
            {scanning ? 'Scanning...' : 'Scan Now'}
          </button>
        </div>

        <div className="grid grid-cols-4 gap-3 text-sm mb-4">
          <div>
            <div className="text-xs text-gray-500">Last Scan</div>
            <div className="text-gray-300">{timeAgo(scannerState?.last_scan) || '—'}</div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Next Scan</div>
            <div className="text-gray-300">{timeUntil(scannerState?.next_scan) || '—'}</div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Scans Today</div>
            <div className="text-gray-300">{scannerState?.scans_today ?? '—'}</div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Bets Placed Today</div>
            <div className="text-purple-300 font-semibold">{scannerState?.bets_placed_today ?? '—'}</div>
          </div>
        </div>

        {/* Live Opportunities */}
        {opportunities.length > 0 ? (
          <div>
            <div className="text-xs text-gray-500 mb-2">Live Opportunities (from last scan)</div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr>
                    {['Player', 'Line', 'Kalshi Price', 'Model Prob', 'Edge', 'Direction', 'Confidence'].map(h => (
                      <th key={h} className="px-2 py-1.5 text-left text-gray-500 uppercase">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {opportunities.map((o, i) => (
                    <tr key={i} className="hover:bg-gray-800">
                      <td className="px-2 py-2 font-medium text-white">{o.player_name}</td>
                      <td className="px-2 py-2 text-gray-400">{o.line}</td>
                      <td className="px-2 py-2 text-gray-300">
                        {o.direction === 'over'
                          ? `${(o.kalshi_over_prob * 100).toFixed(0)}¢`
                          : `${(o.kalshi_under_prob * 100).toFixed(0)}¢`}
                      </td>
                      <td className="px-2 py-2 text-blue-400">{(o.model_prob * 100).toFixed(1)}%</td>
                      <td className={`px-2 py-2 font-semibold ${Math.abs(o.edge) >= 0.10 ? 'text-green-400' : 'text-yellow-400'}`}>
                        {o.edge > 0 ? '+' : ''}{(o.edge * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-2">
                        <span className={`font-medium uppercase ${o.direction === 'over' ? 'text-green-400' : 'text-red-400'}`}>
                          {o.direction}
                        </span>
                      </td>
                      <td className="px-2 py-2">
                        <span className={`px-1 py-0.5 rounded text-xs ${
                          o.confidence === 'high' ? 'bg-green-900 text-green-300' :
                          o.confidence === 'medium' ? 'bg-yellow-900 text-yellow-300' :
                          'bg-gray-800 text-gray-400'
                        }`}>
                          {o.confidence}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="text-xs text-gray-600">
            No opportunities found in last scan. Run pipeline first to generate predictions, then scan.
          </div>
        )}
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
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Set Your Bankroll</h3>
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
              Set Amount
            </button>
          </div>
          <p className="text-xs text-gray-600 mt-2">Enter how much you want to paper trade with. Scanner will size bets from this amount using Kelly criterion.</p>
        </div>

        {/* Settle */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">End of Day Settlement</h3>
          <div className="flex items-center gap-2 flex-wrap">
            <input
              type="date"
              value={betsDate}
              onChange={e => setBetsDate(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-sm text-gray-300 rounded px-2 py-1.5"
            />
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
          <p className="text-xs text-gray-600 mt-2">Pull actual scores in Daily Results first, then settle here to calculate P&L and update the RL agent.</p>
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
                {['Player', 'Direction', 'Amount', 'Source', 'Time', 'Odds', 'Edge', 'Status', 'P&L'].map(h => (
                  <th key={h} className="px-3 py-2 text-left text-xs text-gray-500 uppercase">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {bets.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-3 py-8 text-center text-gray-600">
                    No bets for this date. The scanner will place bets automatically when it finds good Kalshi edges.
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
                  <td className="px-3 py-2.5 text-gray-500 text-xs">{b.scan_time || '—'}</td>
                  <td className="px-3 py-2.5 text-gray-400">{b.decimal_odds?.toFixed(3)}</td>
                  <td className="px-3 py-2.5 text-blue-400">{b.edge != null ? `${(b.edge * 100).toFixed(1)}%` : '—'}</td>
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
