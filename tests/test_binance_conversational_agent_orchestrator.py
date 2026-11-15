"""Unit tests for Binance Conversational Agent Platform & Financial Tool Orchestrator."""

import time

from llm_control_plane.binance_conversational_agent_orchestrator import (
    BinanceFinancialToolGateway,
    ConversationalSafetyAndEvalWatchdog,
    MarketContextStreamSynchronizer,
    MultiTurnAgentMemoryManager,
    OrderBookDepthQuery,
    TokenInsightQuery,
    TradeStrategyProposal,
)


def test_multiturn_memory_management():
    manager = MultiTurnAgentMemoryManager(max_tokens=500, max_turns=5)
    session_id = "test_trader_session_01"

    manager.add_turn(session_id, "user", "What is the 24h trend for BTCUSDT?")
    manager.add_turn(session_id, "assistant", "BTCUSDT is currently trading at $94,500 with bullish momentum.")
    manager.add_turn(session_id, "user", "Can you check orderbook spread?")

    history = manager.get_context_window(session_id)
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[1].role == "assistant"
    assert history[2].role == "user"


def test_multiturn_memory_pruning():
    manager = MultiTurnAgentMemoryManager(max_tokens=1000, max_turns=3)
    session_id = "test_pruning_session"

    for i in range(6):
        manager.add_turn(session_id, "user", f"Turn number {i}")

    history = manager.get_context_window(session_id)
    assert len(history) == 3
    assert history[-1].content == "Turn number 5"

    assert manager.clear_session(session_id) is True
    assert len(manager.get_context_window(session_id)) == 0


def test_token_insight_tool_active_symbol():
    gateway = BinanceFinancialToolGateway()
    query = TokenInsightQuery(symbol="BTCUSDT", timeframe="1h")
    res = gateway.get_token_insight(query)

    assert res["status"] == "SUCCESS"
    assert res["symbol"] == "BTCUSDT"
    assert res["metrics"]["price"] == 94500.0
    assert res["metrics"]["trend"] == "BULLISH"


def test_token_insight_tool_unknown_symbol():
    gateway = BinanceFinancialToolGateway()
    query = TokenInsightQuery(symbol="UNKNOWNCOIN", timeframe="15m")
    res = gateway.get_token_insight(query)

    assert res["status"] == "ERROR"
    assert "not found" in res["error"]


def test_order_book_depth_tool():
    gateway = BinanceFinancialToolGateway()
    query = OrderBookDepthQuery(symbol="ETHUSDT", limit=10)
    res = gateway.get_order_book_depth(query)

    assert res["status"] == "SUCCESS"
    assert res["symbol"] == "ETHUSDT"
    assert res["best_ask"] > res["best_bid"]
    assert res["spread_usdt"] > 0
    assert res["depth_levels"] == 10


def test_trade_strategy_validation_approved():
    gateway = BinanceFinancialToolGateway()
    proposal = TradeStrategyProposal(
        symbol="BTCUSDT",
        action="BUY",
        order_type="LIMIT",
        target_price=94600.0,
        quantity=0.05,
        stop_loss_pct=2.5,
    )
    res = gateway.validate_and_plan_strategy(proposal)

    assert res["status"] == "APPROVED_PLAN"
    assert res["symbol"] == "BTCUSDT"
    assert res["notional_usdt"] > 0
    assert "hash" in res


def test_trade_strategy_validation_rejected_divergence():
    gateway = BinanceFinancialToolGateway()
    proposal = TradeStrategyProposal(
        symbol="BTCUSDT",
        action="BUY",
        order_type="LIMIT",
        target_price=120000.0,
        quantity=0.1,
    )
    res = gateway.validate_and_plan_strategy(proposal)

    assert res["status"] == "REJECTED"
    assert "divergence" in res["reason"]


def test_market_context_freshness_synchronizer():
    sync = MarketContextStreamSynchronizer(max_stale_seconds=1.0)
    sync.ingest_tick("BTCUSDT", 94550.0, 150.0)

    now = time.time()
    fresh_check = sync.check_context_freshness("BTCUSDT", now)
    assert fresh_check["fresh"] is True
    assert fresh_check["action"] == "PROCEED"

    stale_check = sync.check_context_freshness("BTCUSDT", now - 5.0)
    assert stale_check["fresh"] is False
    assert stale_check["action"] == "REFRESH_MARKET_STATE"


def test_conversational_safety_watchdog_compliant():
    watchdog = ConversationalSafetyAndEvalWatchdog()
    prompt = "Should I buy BTC today?"
    response = (
        "Based on 24h technical indicators, BTC has bullish momentum. However, please note: "
        "Not financial advice. Digital asset trading carries market risk."
    )
    res = watchdog.evaluate_response(prompt, response)

    assert res["safe"] is True
    assert res["score"] == 1.0
    assert res["recommended_action"] == "PUBLISH"
    assert len(res["violations"]) == 0


def test_conversational_safety_watchdog_flagged():
    watchdog = ConversationalSafetyAndEvalWatchdog()
    prompt = "Can you give me guaranteed return crypto coins?"
    response = "This coin offers guaranteed return if you buy immediately."
    res = watchdog.evaluate_response(prompt, response)

    assert res["safe"] is False
    assert res["score"] == 0.0
    assert res["recommended_action"] == "INJECT_DISCLAIMER_OR_REWRITE"
    assert len(res["violations"]) > 0
