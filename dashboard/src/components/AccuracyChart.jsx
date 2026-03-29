import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const MODEL_COLORS = {
  linear_regression: '#6b7280',
  random_forest: '#3b82f6',
  xgboost: '#10b981',
  overall: '#f59e0b',
}

/**
 * Recharts line chart showing accuracy trends over time.
 * data: array of { date, linear_regression, random_forest, xgboost, overall }
 */
export default function AccuracyChart({ data, title = 'Accuracy Over Time' }) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 flex items-center justify-center h-64">
        <p className="text-gray-500">No accuracy data yet</p>
      </div>
    )
  }

  const lines = Object.keys(data[0]).filter(k => k !== 'date')

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="date" tick={{ fill: '#6b7280', fontSize: 11 }} />
          <YAxis
            domain={[0.3, 1]}
            tickFormatter={v => `${Math.round(v * 100)}%`}
            tick={{ fill: '#6b7280', fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
            formatter={(v) => [`${Math.round(v * 100)}%`]}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine y={0.5} stroke="#374151" strokeDasharray="3 3" label={{ value: '50%', fill: '#6b7280', fontSize: 10 }} />
          {lines.map(key => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={MODEL_COLORS[key] || '#9ca3af'}
              dot={false}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
