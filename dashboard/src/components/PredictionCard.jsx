import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

function ConfidenceBadge({ level }) {
  const styles = {
    high: 'bg-green-900 text-green-300 border border-green-700',
    medium: 'bg-yellow-900 text-yellow-300 border border-yellow-700',
    low: 'bg-gray-800 text-gray-400 border border-gray-700',
  }
  return (
    <span className={`text-xs px-2 py-0.5 rounded font-medium ${styles[level] || styles.low}`}>
      {level?.toUpperCase()}
    </span>
  )
}

function ProbBar({ label, prob, color }) {
  if (prob == null) return null
  const pct = Math.round(prob * 100)
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-gray-500 w-20 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-1.5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-gray-300 w-8 text-right">{pct}%</span>
    </div>
  )
}

export default function PredictionCard({ pred }) {
  const edge = pred.edge ?? 0
  const EdgeIcon = edge > 0 ? TrendingUp : edge < 0 ? TrendingDown : Minus
  const edgeColor = Math.abs(edge) >= 0.10 ? 'text-green-400' : Math.abs(edge) >= 0.05 ? 'text-yellow-400' : 'text-gray-400'

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="font-semibold text-white">{pred.player_name_display || pred.player_name}</div>
          <div className="text-xs text-gray-500 mt-0.5">Line: {pred.sportsbook_line} pts</div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <ConfidenceBadge level={pred.confidence} />
          <div className={`flex items-center gap-1 text-sm font-bold ${edgeColor}`}>
            <EdgeIcon size={14} />
            {edge >= 0 ? '+' : ''}{(edge * 100).toFixed(1)}% edge
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4 mb-3">
        <div className="text-center">
          <div className="text-2xl font-bold text-white">{Math.round(pred.predicted_points)}</div>
          <div className="text-xs text-gray-500">Predicted</div>
        </div>
        <div className="flex-1">
          <div className="space-y-1.5">
            <ProbBar label="Model" prob={pred.predicted_over_prob} color="bg-blue-500" />
            <ProbBar label="Sportsbook" prob={pred.sportsbook_implied_prob} color="bg-gray-500" />
            {pred.kalshi_prob != null && (
              <ProbBar label="Kalshi" prob={pred.kalshi_prob} color="bg-purple-500" />
            )}
            {pred.polymarket_prob != null && (
              <ProbBar label="Polymarket" prob={pred.polymarket_prob} color="bg-pink-500" />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
