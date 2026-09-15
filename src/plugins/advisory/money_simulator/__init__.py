"""The money simulator: a funded paper book scored in rupees.

Not a strategy and not a forecast — a *consequence*. Everything else in the
advisory layer says what a leg is worth doing; this says what it costs to have
done it, at a fixed lot, against the tape.

The pack is deliberately small and deeply split: wallets and the reserve in
:mod:`~plugins.advisory.money_simulator.wallet`, one open lot in
:mod:`~plugins.advisory.money_simulator.position`, the blend of every signal into
a single view in :mod:`~plugins.advisory.money_simulator.policy`, the decision
loop in :mod:`~plugins.advisory.money_simulator.simulator`, and its memory in
:mod:`~plugins.advisory.money_simulator.ledger`.
"""

from .ledger import ClosedTrade, SimulationLedger
from .policy import (
    BUY,
    EXIT,
    HOLD,
    SELL,
    Decision,
    DecisionPolicy,
    LegState,
    LegView,
    PolicyWeights,
)
from .position import Position
from .simulator import MoneySimulator, SimulatorConfig
from .wallet import Reserve, Wallet

__all__ = [
    "BUY",
    "EXIT",
    "HOLD",
    "SELL",
    "ClosedTrade",
    "Decision",
    "DecisionPolicy",
    "LegState",
    "LegView",
    "MoneySimulator",
    "PolicyWeights",
    "Position",
    "Reserve",
    "SimulationLedger",
    "SimulatorConfig",
    "Wallet",
]
