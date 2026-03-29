import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { TrendingUp, BarChart2, GitCompare, Activity, DollarSign, CheckCircle2 } from 'lucide-react'
import TodaysPredictions from './pages/TodaysPredictions.jsx'
import ModelPerformance from './pages/ModelPerformance.jsx'
import MarketComparison from './pages/MarketComparison.jsx'
import PaperTrader from './pages/PaperTrader.jsx'
import DailyResults from './pages/DailyResults.jsx'

const NAV = [
  { to: '/', label: "Today's Picks", Icon: TrendingUp },
  { to: '/results', label: 'Daily Results', Icon: CheckCircle2 },
  { to: '/performance', label: 'Model Performance', Icon: BarChart2 },
  { to: '/comparison', label: 'Market Comparison', Icon: GitCompare },
  { to: '/trader', label: 'Paper Trader', Icon: DollarSign },
]

function NavBar() {
  return (
    <header className="border-b border-gray-800 bg-gray-900 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 flex items-center gap-8 h-14">
        <div className="flex items-center gap-2 text-blue-400 font-bold text-lg tracking-tight">
          <Activity size={20} />
          NBA Predictor
        </div>
        <nav className="flex gap-1">
          {NAV.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                }`
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main className="max-w-7xl mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<TodaysPredictions />} />
          <Route path="/performance" element={<ModelPerformance />} />
          <Route path="/comparison" element={<MarketComparison />} />
          <Route path="/trader" element={<PaperTrader />} />
          <Route path="/results" element={<DailyResults />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
