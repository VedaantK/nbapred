"""
Q-learning reinforcement learning agent for paper bet sizing.

State space (81 states):
  - edge_bucket:      low (0.05-0.10) | medium (0.10-0.20) | high (>0.20)    → 3
  - confidence:       low | medium | high                                      → 3
  - recent_accuracy:  poor (<0.45) | average (0.45-0.55) | good (>0.55)      → 3
  - time_bucket:      morning (<12pm) | afternoon (12-6pm) | evening (>6pm)  → 3
  (total: 3 × 3 × 3 × 3 = 81 states)

Actions (5):
  0 = skip bet
  1 = quarter Kelly
  2 = half Kelly  (default)
  3 = three-quarter Kelly
  4 = full Kelly

The Q-table is a flat dict keyed by state strings.
Persisted as JSON so it survives server restarts.
"""

import json
import random
from pathlib import Path

from config.settings import MODELS_DIR
from config.logging_config import setup_logging

logger = setup_logging("rl_agent")

QTABLE_PATH = Path(MODELS_DIR) / "rl_qtable.json"

# Hyperparameters
ALPHA = 0.1          # learning rate
GAMMA = 0.9          # discount factor (single-step, so mainly for future-proofing)
EPSILON_START = 0.3  # exploration rate at cold-start
EPSILON_MIN = 0.05   # minimum exploration
EPSILON_DECAY = 0.995

# Kelly multipliers per action
ACTION_KELLY_MULT = {
    0: 0.0,    # skip
    1: 0.25,   # quarter Kelly
    2: 0.5,    # half Kelly (default baseline)
    3: 0.75,   # three-quarter Kelly
    4: 1.0,    # full Kelly
}
N_ACTIONS = len(ACTION_KELLY_MULT)


def _edge_bucket(edge: float) -> str:
    abs_edge = abs(edge)
    if abs_edge >= 0.20:
        return "high"
    if abs_edge >= 0.10:
        return "medium"
    return "low"


def _accuracy_bucket(recent_accuracy: float | None) -> str:
    if recent_accuracy is None:
        return "average"
    if recent_accuracy > 0.55:
        return "good"
    if recent_accuracy >= 0.45:
        return "average"
    return "poor"


def encode_state(
    edge: float,
    confidence: str,
    recent_accuracy: float | None,
    time_bucket: str = "afternoon",
) -> str:
    """Return a compact string key for the Q-table state."""
    eb = _edge_bucket(edge)
    ab = _accuracy_bucket(recent_accuracy)
    return f"{eb}|{confidence}|{ab}|{time_bucket}"


class RLBettingAgent:
    """
    Epsilon-greedy Q-learning agent that learns optimal Kelly multipliers.

    Usage:
        agent = RLBettingAgent.load()
        action = agent.choose_action(state_key)
        kelly_mult = ACTION_KELLY_MULT[action]
        # ... place bet, get P&L ...
        agent.update(state_key, action, reward, next_state_key)
        agent.save()
    """

    def __init__(self):
        self.q: dict[str, list[float]] = {}   # state → [Q(a0), Q(a1), ..., Q(a4)]
        self.epsilon = EPSILON_START
        self.total_updates = 0

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self):
        QTABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q": self.q,
            "epsilon": round(self.epsilon, 6),
            "total_updates": self.total_updates,
        }
        with open(QTABLE_PATH, "w") as f:
            json.dump(data, f)
        logger.debug(f"RL agent saved ({len(self.q)} states, ε={self.epsilon:.4f})")

    @classmethod
    def load(cls) -> "RLBettingAgent":
        agent = cls()
        if QTABLE_PATH.exists():
            try:
                with open(QTABLE_PATH) as f:
                    data = json.load(f)
                agent.q = data.get("q", {})
                agent.epsilon = data.get("epsilon", EPSILON_START)
                agent.total_updates = data.get("total_updates", 0)
                logger.info(f"RL agent loaded ({len(agent.q)} states, ε={agent.epsilon:.4f})")
            except Exception as e:
                logger.warning(f"Failed to load RL Q-table, starting fresh: {e}")
        return agent

    # ── Q-table access ────────────────────────────────────────────────────────

    def _get_q(self, state: str) -> list[float]:
        if state not in self.q:
            # Initialize with slight preference for half-Kelly (action 2)
            self.q[state] = [0.0, 0.0, 0.05, 0.0, -0.05]
        return self.q[state]

    # ── decision ─────────────────────────────────────────────────────────────

    def choose_action(self, state: str) -> int:
        """Epsilon-greedy action selection. Returns action index 0-4."""
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        q_vals = self._get_q(state)
        return int(q_vals.index(max(q_vals)))

    def get_kelly_multiplier(
        self,
        edge: float,
        confidence: str,
        recent_accuracy: float | None,
        time_bucket: str = "afternoon",
    ) -> float:
        """
        Convenience method: encode state, choose action, return Kelly multiplier.
        Skips exploration randomness in inference mode (epsilon=0 temporarily).
        """
        state = encode_state(edge, confidence, recent_accuracy, time_bucket)
        q_vals = self._get_q(state)
        action = int(q_vals.index(max(q_vals)))
        return ACTION_KELLY_MULT[action]

    # ── learning ─────────────────────────────────────────────────────────────

    def update(self, state: str, action: int, reward: float, next_state: str):
        """
        Single Q-learning update:
          Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        Reward is normalized P&L (profit_loss / bet_amount) to make it
        scale-independent across different bet sizes.
        """
        q_sa = self._get_q(state)[action]
        q_next_max = max(self._get_q(next_state))
        td_target = reward + GAMMA * q_next_max
        td_error = td_target - q_sa
        self.q[state][action] = round(q_sa + ALPHA * td_error, 6)
        self.total_updates += 1

        # Decay exploration
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def get_stats(self) -> dict:
        """Return diagnostic stats for the /api/trader/rl-status endpoint."""
        action_counts = {i: 0 for i in range(N_ACTIONS)}
        greedy_actions = {}
        for state, q_vals in self.q.items():
            best = int(q_vals.index(max(q_vals)))
            greedy_actions[state] = best
            action_counts[best] += 1

        return {
            "states_explored": len(self.q),
            "total_updates": self.total_updates,
            "epsilon": round(self.epsilon, 4),
            "greedy_action_distribution": action_counts,
            "greedy_actions_by_state": greedy_actions,
            "action_labels": {
                0: "skip",
                1: "quarter_kelly",
                2: "half_kelly",
                3: "three_quarter_kelly",
                4: "full_kelly",
            },
        }
