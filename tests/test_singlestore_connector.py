"""Tests for SingleStore distributed vector & hybrid search MCP connector."""

import threading

import pytest

from llm_control_plane import (
    ConnectorRegistry,
    SingleStoreConfig,
    SingleStoreConnectionPool,
    SingleStoreMCPConnector,
    SingleStorePoolExhaustedError,
    SingleStoreQueryResult,
    SingleStoreSecurityError,
    ToolCallRequest,
)


def test_singlestore_config_defaults() -> None:
    cfg = SingleStoreConfig()
    assert cfg.port == 3306
    assert cfg.database == "singlestore_ai"
    assert cfg.pool_size == 10
    assert cfg.max_overflow == 20
    assert cfg.enforce_read_only is True


def test_connection_pool_lifecycle_and_exhaustion() -> None:
    cfg = SingleStoreConfig(pool_size=2, max_overflow=1)
    pool = SingleStoreConnectionPool(cfg)

    assert pool.saturation_ratio == 0.0

    conn1 = pool.lease_connection()
    conn2 = pool.lease_connection()
    assert conn2 is not None
    assert pool.saturation_ratio == pytest.approx(2 / 3, 0.01)

    # Overflow connection
    conn3 = pool.lease_connection()
    assert conn3 is not None
    assert pool.saturation_ratio == 1.0

    # Exhausted
    with pytest.raises(SingleStorePoolExhaustedError):
        pool.lease_connection()

    pool.release_connection(conn1)
    assert pool.saturation_ratio == pytest.approx(2 / 3, 0.01)

    # Re-lease after return
    conn4 = pool.lease_connection()
    assert conn4 is not None


def test_sql_execution_and_read_only_security() -> None:
    connector = SingleStoreMCPConnector()
    mock_data = [
        {"id": 1, "name": "vector_doc_1"},
        {"id": 2, "name": "vector_doc_2"},
        {"id": 3, "name": "vector_doc_3"},
    ]

    # Valid SELECT
    res = connector.execute_sql("SELECT * FROM documents", limit=2, mock_data=mock_data)
    assert isinstance(res, SingleStoreQueryResult)
    assert res.row_count == 2
    assert len(res.rows) == 2
    assert res.execution_time_ms >= 0.0

    # Prohibited write operations
    with pytest.raises(SingleStoreSecurityError):
        connector.execute_sql("DROP TABLE documents")

    with pytest.raises(SingleStoreSecurityError):
        connector.execute_sql("DELETE FROM documents WHERE id = 1")

    with pytest.raises(SingleStoreSecurityError):
        connector.execute_sql("TRUNCATE TABLE documents")


def test_vector_similarity_search() -> None:
    connector = SingleStoreMCPConnector()
    table_name = "knowledge_embeddings"

    # Seed 3 records with embeddings
    records = [
        {"id": "doc-a", "text": "Kubernetes cluster architecture", "embedding": [1.0, 0.0, 0.0]},
        {"id": "doc-b", "text": "SingleStore distributed SQL", "embedding": [0.8, 0.6, 0.0]},
        {"id": "doc-c", "text": "Unrelated topic", "embedding": [0.0, 1.0, 0.0]},
    ]
    connector.set_mock_table_data(table_name, records)

    # Search with query vector pointing towards doc-a
    res = connector.vector_search(
        table_name=table_name,
        query_vector=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert res.row_count == 2
    assert res.rows[0]["id"] == "doc-a"
    assert res.rows[0]["similarity_score"] == 1.0
    assert res.rows[1]["id"] == "doc-b"
    assert res.rows[1]["similarity_score"] == pytest.approx(0.8, 0.01)


def test_mcp_tool_registration_and_execution() -> None:
    registry = ConnectorRegistry()
    connector = SingleStoreMCPConnector()
    connector.set_mock_table_data(
        "metrics_store",
        [{"metric": "cpu_util", "embedding": [0.5, 0.5]}],
    )

    connector.register_tools(registry)

    # Verify tool specs registered
    spec_names = {t.name for t in registry.list_specs()}
    assert "singlestore_execute_sql" in spec_names
    assert "singlestore_vector_search" in spec_names
    assert "singlestore_table_schema" in spec_names

    # 1. Execute SQL tool
    req_sql = ToolCallRequest(
        tenant_id="tenant-sre",
        tool_name="singlestore_execute_sql",
        arguments={"sql": "SELECT * FROM metrics_store", "limit": 10},
    )
    result_sql = registry.execute(req_sql)
    assert result_sql.status == "executed"
    assert "row_count" in result_sql.output

    # 2. Execute Vector Search tool
    req_vec = ToolCallRequest(
        tenant_id="tenant-sre",
        tool_name="singlestore_vector_search",
        arguments={"table_name": "metrics_store", "query_vector": [0.5, 0.5], "top_k": 1},
    )
    result_vec = registry.execute(req_vec)
    assert result_vec.status == "executed"
    assert result_vec.output["top_k"] == 1

    # 3. Execute Schema tool
    req_schema = ToolCallRequest(
        tenant_id="tenant-sre",
        tool_name="singlestore_table_schema",
        arguments={"table_name": "metrics_store"},
    )
    result_schema = registry.execute(req_schema)
    assert result_schema.status == "executed"
    assert result_schema.output["engine"] == "SingleStore Columnstore"


def test_singlestore_connector_concurrency() -> None:
    connector = SingleStoreMCPConnector(SingleStoreConfig(pool_size=10, max_overflow=10))
    threads = []
    errors = []

    def worker(idx: int) -> None:
        try:
            res = connector.execute_sql(f"SELECT {idx}", mock_data=[{"v": idx}])
            assert res.row_count == 1
        except Exception as e:
            errors.append(e)

    for i in range(20):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0
