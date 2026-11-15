"""Unit tests for Starburst AI Platform Bridge & Developer Primitives SDK."""

from __future__ import annotations

import pytest

from llm_control_plane.starburst_ai_platform_bridge_sdk import (
    CrossLanguageExceptionTranslator,
    DeterministicToolExecutionGateway,
    InteropMessageEnvelope,
    JVMDataType,
    ResilientStreamingMultiplexer,
    StarburstAIPlatformSDK,
    StreamTokenChunk,
    TokenBucketQuotaManager,
    ToolExecutionContract,
)


def test_interop_envelope_create_and_decode():
    data = {"query_id": "20261115_trino_01", "partitions": ["p1", "p2"], "row_count": 10500}
    envelope = InteropMessageEnvelope.create(
        message_id="msg-101",
        payload_type="trino.lakehouse.PartitionStats",
        jvm_type=JVMDataType.MAP,
        data=data,
        metadata={"coordinator": "trino-coord-01.starburst.internal"},
    )

    assert envelope.message_id == "msg-101"
    assert envelope.jvm_type == JVMDataType.MAP
    assert envelope.metadata["coordinator"] == "trino-coord-01.starburst.internal"
    assert envelope.checksum_crc32 != 0

    decoded = envelope.verify_and_decode()
    assert decoded["query_id"] == "20261115_trino_01"
    assert decoded["partitions"] == ["p1", "p2"]
    assert decoded["row_count"] == 10500


def test_interop_envelope_checksum_tampering_detection():
    data = {"status": "HEALTHY"}
    envelope = InteropMessageEnvelope.create(
        message_id="msg-tamper",
        payload_type="trino.health.Status",
        jvm_type=JVMDataType.STRING,
        data=data,
    )

    # Tamper with byte payload
    envelope.payload_bytes = b'{"status":"CORRUPTED"}'

    with pytest.raises(ValueError, match="Checksum mismatch across JVM boundary"):
        envelope.verify_and_decode()


def test_cross_language_exception_translator_py_to_jvm():
    val_err = ValueError("Invalid partition specification")
    jvm_payload = CrossLanguageExceptionTranslator.to_jvm_error_payload(val_err)
    assert jvm_payload["error_class"] == "java.lang.IllegalArgumentException"
    assert "Invalid partition specification" in jvm_payload["message"]

    timeout_err = TimeoutError("gRPC socket deadline exceeded")
    jvm_timeout = CrossLanguageExceptionTranslator.to_jvm_error_payload(timeout_err)
    assert jvm_timeout["error_class"] == "java.util.concurrent.TimeoutException"

    perm_err = PermissionError("Catalog access denied for tenant data-science")
    jvm_sec = CrossLanguageExceptionTranslator.to_jvm_error_payload(perm_err)
    assert jvm_sec["error_class"] == "java.lang.SecurityException"


def test_cross_language_exception_translator_jvm_to_py():
    jvm_payload = {
        "error_class": "io.trino.spi.TrinoException",
        "message": "SYNTAX_ERROR: line 1:15: Table 'lakehouse.default.missing' does not exist",
    }
    py_exc = CrossLanguageExceptionTranslator.from_jvm_error_payload(jvm_payload)
    assert isinstance(py_exc, RuntimeError)
    assert "Table 'lakehouse.default.missing' does not exist" in str(py_exc)

    jvm_cast_payload = {
        "error_class": "java.lang.ClassCastException",
        "message": "Cannot cast VARCHAR to BIGINT",
    }
    py_cast_exc = CrossLanguageExceptionTranslator.from_jvm_error_payload(jvm_cast_payload)
    assert isinstance(py_cast_exc, TypeError)


def test_token_bucket_quota_manager():
    mgr = TokenBucketQuotaManager(default_rate_per_sec=50.0, default_burst_capacity=100.0)
    tenant = "analytics-agent"

    # Consume within burst capacity
    assert mgr.consume(tenant, 80) is True

    # Immediate excessive request should fail
    assert mgr.consume(tenant, 50) is False

    # Check available tokens
    available = mgr.get_available_tokens(tenant)
    assert available < 50.0

    # Configure dedicated high-throughput tenant
    mgr.configure_tenant("vip-agent", rate=500.0, capacity=1000.0)
    assert mgr.consume("vip-agent", 500) is True


def test_resilient_streaming_multiplexer():
    mux = ResilientStreamingMultiplexer(buffer_size=5)
    stream_id = "stream-eval-01"

    for i in range(7):
        chunk = StreamTokenChunk(chunk_index=i, text=f"token_{i}", latency_ms=1.0)
        mux.push_chunk(stream_id, chunk)

    # Chunks 0 and 1 should have been evicted by buffer_size=5
    reconnected_chunks = mux.get_chunks_from(stream_id, from_index=3)
    indices = [c.chunk_index for c in reconnected_chunks]
    assert indices == [3, 4, 5, 6]

    # Non-existent stream returns empty list
    assert mux.get_chunks_from("missing-stream", 0) == []


def test_tool_gateway_validation_and_safety():
    gateway = DeterministicToolExecutionGateway()

    # Unauthorized catalog
    bad_catalog = ToolExecutionContract(
        tool_name="trino_query_catalog",
        arguments={"sql": "SELECT 1"},
        catalog="untrusted_external_db",
    )
    res_bad = gateway.validate_tool_call(bad_catalog)
    assert res_bad.valid is False
    assert "unauthorized" in res_bad.error_message

    # Destructive SQL rejection
    drop_contract = ToolExecutionContract(
        tool_name="trino_query_catalog",
        arguments={"sql": "DROP TABLE lakehouse.sales.orders"},
        catalog="lakehouse",
    )
    res_drop = gateway.validate_tool_call(drop_contract)
    assert res_drop.valid is False
    assert "Destructive queries" in res_drop.error_message

    # Missing limit auto-repair
    no_limit = ToolExecutionContract(
        tool_name="trino_query_catalog",
        arguments={"sql": "SELECT id, amount FROM lakehouse.sales.orders"},
        catalog="lakehouse",
    )
    res_valid = gateway.validate_tool_call(no_limit)
    assert res_valid.valid is True
    assert "LIMIT 1000" in res_valid.sanitized_sql


def test_tool_gateway_execute_with_repair_loop():
    gateway = DeterministicToolExecutionGateway()

    # Initial invalid catalog gets repaired to 'lakehouse'
    contract = ToolExecutionContract(
        tool_name="trino_query_catalog",
        arguments={"sql": "SELECT count(*) FROM lakehouse.default.metrics"},
        catalog="invalid_cat",
    )

    executed = gateway.execute_with_repair_loop(
        contract,
        executor_fn=lambda q: {"status": "OK", "rows": 1, "sql": q},
    )

    assert executed["success"] is True
    assert executed["attempts"] == 2
    assert "LIMIT 1000" in executed["sanitized_sql"]


def test_starburst_ai_platform_sdk_facade():
    sdk = StarburstAIPlatformSDK(cluster_endpoint="trino.starburst.internal")

    # JVM Bridge call
    envelope = sdk.call_jvm_bridge(
        payload_name="lakehouse.VectorSearchConfig",
        jvm_type=JVMDataType.MAP,
        data={"top_k": 5, "similarity_metric": "COSINE"},
    )
    assert envelope.jvm_type == JVMDataType.MAP
    assert envelope.verify_and_decode()["top_k"] == 5

    # Model invocation within budget
    res = sdk.request_model_invocation("team-alpha", "Generate semantic catalog summary", 40)
    assert res["status"] == "COMPLETED"

    # Streaming tokens
    tokens = ["SELECT", " ", "COUNT(*)", " ", "FROM"]
    stream = list(sdk.stream_tokens("str-01", tokens))
    assert len(stream) == 5
    assert stream[-1].is_terminal is True

    # Lakehouse tool execution
    tool_res = sdk.execute_lakehouse_tool(
        tool_name="trino_describe_table",
        arguments={"table_name": "customer_360"},
        catalog="iceberg",
    )
    assert tool_res["success"] is True
    assert "DESCRIBE iceberg.default.customer_360" in tool_res["sanitized_sql"]

    assert sdk.metrics["total_requests"] >= 1
    assert sdk.metrics["jvm_bridges_called"] >= 1
    assert sdk.metrics["tools_executed"] >= 1
