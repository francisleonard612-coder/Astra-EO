from __future__ import annotations

import time
from dataclasses import dataclass

from app.logging_setup import get_logger
from decision.decision_engine import Decision
from ingestion.deriv_client import DerivClient, DerivRequestError
from pricing.payout import ContractQuote, get_quote

logger = get_logger("execution.orders")

MAX_QUOTE_AGE_SECONDS = 2.0


@dataclass
class TradeResult:
    symbol: str
    contract_type: str  # DIGITEVEN | DIGITODD
    stake: float
    payout: float
    contract_id: int | None
    won: bool | None
    pnl: float | None
    error: str | None = None


async def execute_decision(client: DerivClient, decision: Decision, currency: str,
                            duration: int, duration_unit: str, dry_run: bool) -> TradeResult | None:
    if decision.decision == "NO_TRADE" or decision.stake is None:
        return None

    side = decision.decision.split("_", 1)[1]  # "TRADE_EVEN" -> "EVEN", "TRADE_ODD" -> "ODD"
    contract_type = "DIGITEVEN" if side == "EVEN" else "DIGITODD"
    quote = decision.quote_even if side == "EVEN" else decision.quote_odd

    if quote is None:
        return TradeResult(decision.symbol, contract_type, decision.stake, 0.0, None, None, None,
                            error="no_quote_at_decision_time")

    # Re-validate: fetch a fresh quote immediately before buying and refuse to
    # trade on stale contract information (spec section 18).
    age_check_start = time.time()
    fresh_quote = await get_quote(client, decision.symbol, contract_type, decision.stake,
                                   duration, duration_unit, currency)
    if fresh_quote is None:
        return TradeResult(decision.symbol, contract_type, decision.stake, 0.0, None, None, None,
                            error="quote_unavailable_at_execution")

    payout_drift = abs(fresh_quote.payout - quote.payout) / max(quote.payout, 1e-9)
    if payout_drift > 0.15:
        logger.warning("Payout drifted too much between decision and execution, skipping",
                        extra={"extra_fields": {"symbol": decision.symbol, "drift": payout_drift}})
        return TradeResult(decision.symbol, contract_type, decision.stake, fresh_quote.payout,
                            None, None, None, error="stale_quote_payout_drift")

    if dry_run:
        logger.info("DRY_RUN: would execute trade", extra={"extra_fields": {
            "symbol": decision.symbol, "contract_type": contract_type,
            "stake": decision.stake, "payout": fresh_quote.payout,
        }})
        return TradeResult(decision.symbol, contract_type, decision.stake, fresh_quote.payout,
                            None, None, 0.0, error=None)

    try:
        buy_resp = await client.buy(fresh_quote.proposal_id, fresh_quote.ask_price)
    except DerivRequestError as exc:
        logger.error("Buy failed", extra={"extra_fields": {"symbol": decision.symbol, "error": str(exc)}})
        return TradeResult(decision.symbol, contract_type, decision.stake, fresh_quote.payout,
                            None, None, None, error=str(exc))

    contract_id = buy_resp.get("contract_id")
    if contract_id is None:
        return TradeResult(decision.symbol, contract_type, decision.stake, fresh_quote.payout,
                            None, None, None, error="buy_response_missing_contract_id")

    settled = await client.wait_for_contract_settlement(contract_id, timeout=30.0)
    profit = settled.get("profit")
    won = None
    pnl = None
    if profit is not None:
        pnl = float(profit)
        won = pnl > 0

    return TradeResult(
        symbol=decision.symbol, contract_type=contract_type,
        stake=decision.stake, payout=fresh_quote.payout, contract_id=contract_id,
        won=won, pnl=pnl,
    )
