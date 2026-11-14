"""Unit tests for Intercom Fin AI Agent Orchestrator."""

import time

from llm_control_plane.intercom_fin_orchestrator import (
    FinConversationState,
    FinConversationStateMachine,
    FinHybridRAGRouter,
    FinMultiProviderAdaptiveGateway,
    FinProcedureToolExecutor,
    KnowledgeSnippet,
)


def test_conversation_state_machine_flow():
    fsm = FinConversationStateMachine(tenant_id="tenant_123", conversation_id="conv_abc")
    assert fsm.current_state == FinConversationState.INITIAL

    # Informational query
    state = fsm.add_user_message("How do I update my billing email?")
    assert state == FinConversationState.KNOWLEDGE_RETRIEVAL
    assert fsm.intent == "INFORMATIONAL_QA"

    # Procedural query
    state = fsm.add_user_message("I need to process a refund for my order")
    assert state == FinConversationState.PROCEDURAL_ACTION
    assert fsm.intent == "PROCEDURAL_TRANSACTION"

    # Human request
    state = fsm.add_user_message("I want to speak to a human representative please")
    assert state == FinConversationState.HUMAN_ESCALATION
    assert fsm.escalation_reason == "USER_REQUESTED_HUMAN"


def test_conversation_max_turns_escalation():
    fsm = FinConversationStateMachine(tenant_id="tenant_123", conversation_id="conv_abc", max_turns=3)
    fsm.add_user_message("Hello")
    fsm.add_user_message("Are you there?")
    fsm.add_user_message("Still need help")
    # 4th message should trigger escalation
    state = fsm.add_user_message("One more thing")
    assert state == FinConversationState.HUMAN_ESCALATION
    assert fsm.escalation_reason == "MAX_TURNS_EXCEEDED"


def test_hybrid_rag_router_scoring_and_recency():
    router = FinHybridRAGRouter(recency_half_life_days=30.0)
    now = time.time()

    # Create fresh snippet vs stale snippet
    fresh_snippet = KnowledgeSnippet(
        doc_id="doc_fresh",
        title="2026 Refund Policy",
        content="Refunds can be requested within 30 days of purchase for full store credit or card reimbursement.",
        updated_at_epoch=now - 86400 * 2,  # 2 days old
        base_dense_score=0.88,
        base_sparse_score=0.85,
    )
    stale_snippet = KnowledgeSnippet(
        doc_id="doc_stale",
        title="Legacy 2022 Refund Policy",
        content="Refunds can be requested within 14 days of purchase under legacy terms.",
        updated_at_epoch=now - 86400 * 180,  # 180 days old
        base_dense_score=0.89,  # High dense similarity but outdated
        base_sparse_score=0.82,
    )

    scored = router.score_snippets("refund policy days", [stale_snippet, fresh_snippet], current_epoch=now)
    assert len(scored) == 2
    # Fresh snippet should rank first due to recency decay penalty on the 180-day-old doc
    assert scored[0][0].doc_id == "doc_fresh"
    assert scored[0][1] > scored[1][1]


def test_grounding_verification():
    router = FinHybridRAGRouter(grounding_threshold=0.70)
    now = time.time()
    snippet = KnowledgeSnippet(
        doc_id="kb_pricing",
        title="Pricing Guide",
        content="Enterprise plans include unlimited automated agent resolutions and custom SLA coverage.",
        updated_at_epoch=now,
        base_dense_score=0.9,
        base_sparse_score=0.9,
    )

    # Grounded answer
    grounded_ans = (
        "Enterprise plans include unlimited automated agent resolutions. They also feature custom SLA coverage."
    )
    res = router.verify_grounding(grounded_ans, [snippet])
    assert res.is_grounded is True
    assert res.grounding_score >= 0.70
    assert "kb_pricing" in res.cited_doc_ids

    # Ungrounded hallucination
    hallucinated_ans = "We offer free titanium quantum computers with every annual membership."
    res_fake = router.verify_grounding(hallucinated_ans, [snippet])
    assert res_fake.is_grounded is False
    assert len(res_fake.unsupported_claims) > 0


def test_procedure_tool_executor_idempotency_and_validation():
    executor = FinProcedureToolExecutor()

    # 1. Successful order lookup
    res_lookup = executor.execute("order_lookup", {"order_id": "ord_9981"})
    assert res_lookup.success is True
    assert res_lookup.data["status"] == "DELIVERED"
    assert res_lookup.data["eligible_for_refund"] is True

    # 2. Validation failure (missing order_id)
    res_err = executor.execute("order_lookup", {})
    assert res_err.success is False
    assert "Missing required parameter" in (res_err.error_message or "")

    # 3. Idempotent execution
    idempotency_key = "idem_refund_123"
    res1 = executor.execute(
        "process_refund", {"order_id": "ord_9981", "amount_cents": 8900}, idempotency_key=idempotency_key
    )
    assert res1.success is True
    refund_id = res1.data["refund_id"]

    # Re-executing with the same idempotency key returns the exact cached result
    res2 = executor.execute(
        "process_refund", {"order_id": "ord_9981", "amount_cents": 8900}, idempotency_key=idempotency_key
    )
    assert res2.success is True
    assert res2.data["refund_id"] == refund_id


def test_adaptive_gateway_failover_and_quota():
    gateway = FinMultiProviderAdaptiveGateway(tenant_token_limits={"tenant_small": 1000})

    # Within quota
    res1 = gateway.dispatch_inference("tenant_small", "Test prompt", estimated_tokens=400)
    assert res1["success"] is True
    assert res1["provider_used"] == "anthropic_claude_35"

    # Simulate primary outage
    gateway.set_provider_status("anthropic_claude_35", False)
    res2 = gateway.dispatch_inference("tenant_small", "Second prompt", estimated_tokens=400)
    assert res2["success"] is True
    assert res2["provider_used"] == "openai_gpt4o"  # Fallback to secondary

    # Quota exhaustion
    res3 = gateway.dispatch_inference("tenant_small", "Third prompt", estimated_tokens=500)
    assert res3["success"] is False
    assert res3["error"] == "TENANT_QUOTA_EXCEEDED"
