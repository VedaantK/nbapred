/**
 * Side-by-side probability comparison bar for a single player.
 * Shows model vs sportsbook vs prediction markets.
 */
export default function OddsComparison({ modelProb, sbProb, kalshiProb, polyProb, line }) {
  const sources = [
    { label: 'Model', prob: modelProb, color: '#3b82f6' },
    { label: 'Sportsbook', prob: sbProb, color: '#6b7280' },
    { label: 'Kalshi', prob: kalshiProb, color: '#a855f7' },
    { label: 'Polymarket', prob: polyProb, color: '#ec4899' },
  ].filter(s => s.prob != null)

  return (
    <div className="space-y-2">
      {line != null && (
        <div className="text-xs text-gray-500 mb-1">Line: {line} pts — P(Over)</div>
      )}
      {sources.map(({ label, prob, color }) => {
        const pct = Math.round(prob * 100)
        return (
          <div key={label} className="flex items-center gap-2 text-sm">
            <span className="text-gray-400 w-24 shrink-0">{label}</span>
            <div className="flex-1 bg-gray-800 rounded h-2 overflow-hidden">
              <div
                className="h-full rounded transition-all"
                style={{ width: `${pct}%`, backgroundColor: color }}
              />
            </div>
            <span className="text-gray-200 w-10 text-right font-mono">{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}
