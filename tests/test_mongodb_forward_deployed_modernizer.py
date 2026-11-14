"""Unit tests for MongoDB Forward Deployed AI Modernizer & RAG Orchestration Engine."""

import pytest
from llm_control_plane.mongodb_forward_deployed_modernizer import (
    AtlasVectorSearchRAGOrchestrator,
    DocumentModelingStrategy,
    DualQueryConsistencyValidator,
    MongoDBDocumentSchemaSynthesizer,
    RelationalSchemaAnalyzer,
    SQLColumn,
    SQLDataType,
    SQLForeignKey,
    SQLTable,
    StoredProcedureAggregationTranspiler,
)


def test_relational_schema_analyzer_ddl_parsing():
    analyzer = RelationalSchemaAnalyzer()
    ddl = """
    CREATE TABLE orders (
        order_id BIGINT PRIMARY KEY,
        customer_id VARCHAR(64) NOT NULL,
        total_amount DECIMAL(12, 2) NOT NULL,
        created_at TIMESTAMP NOT NULL,
        is_fulfilled BOOLEAN,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );
    """
    table = analyzer.parse_table_ddl(ddl)
    assert table.name == "orders"
    assert len(table.columns) == 5
    assert len(table.foreign_keys) == 1

    col_map = {c.name: c for c in table.columns}
    assert col_map["order_id"].data_type == SQLDataType.BIGINT
    assert col_map["order_id"].is_primary_key is True
    assert col_map["customer_id"].is_nullable is False
    assert col_map["total_amount"].data_type == SQLDataType.DECIMAL
    assert col_map["is_fulfilled"].data_type == SQLDataType.BOOLEAN

    fk = table.foreign_keys[0]
    assert fk.column_name == "customer_id"
    assert fk.target_table == "customers"
    assert fk.target_column == "id"


def test_document_modeling_strategy_heuristics():
    analyzer = RelationalSchemaAnalyzer()
    parent = SQLTable(name="orders", columns=[])
    child_items = SQLTable(name="order_items", columns=[], estimated_cardinality=5)
    child_logs = SQLTable(name="audit_logs", columns=[], estimated_cardinality=10000)

    fk_items = SQLForeignKey("order_id", "orders", "order_id", is_unbounded=False)
    fk_logs = SQLForeignKey("order_id", "orders", "order_id", is_unbounded=True)

    strategy_items = analyzer.determine_strategy(parent, child_items, fk_items)
    strategy_logs = analyzer.determine_strategy(parent, child_logs, fk_logs)

    assert strategy_items == DocumentModelingStrategy.EMBED
    assert strategy_logs == DocumentModelingStrategy.REFERENCE


def test_mongodb_json_schema_synthesizer():
    synthesizer = MongoDBDocumentSchemaSynthesizer()

    parent_table = SQLTable(
        name="customers",
        columns=[
            SQLColumn("id", SQLDataType.VARCHAR, is_primary_key=True, is_nullable=False),
            SQLColumn("email", SQLDataType.VARCHAR, is_nullable=False),
            SQLColumn("loyalty_points", SQLDataType.INT, is_nullable=True),
        ],
    )
    child_table = SQLTable(
        name="addresses",
        columns=[
            SQLColumn("customer_id", SQLDataType.VARCHAR, is_nullable=False),
            SQLColumn("street", SQLDataType.VARCHAR, is_nullable=False),
            SQLColumn("city", SQLDataType.VARCHAR, is_nullable=False),
            SQLColumn("postal_code", SQLDataType.VARCHAR, is_nullable=False),
        ],
    )
    fk = SQLForeignKey("customer_id", "customers", "id", is_unique=False)

    schema = synthesizer.generate_json_schema(parent_table, embedded_tables=[(child_table, fk)])
    assert "$jsonSchema" in schema
    json_schema = schema["$jsonSchema"]

    assert json_schema["bsonType"] == "object"
    assert "_id" in json_schema["required"]
    assert "email" in json_schema["required"]

    props = json_schema["properties"]
    assert props["_id"]["bsonType"] == "objectId"
    assert props["email"]["bsonType"] == "string"
    assert props["loyalty_points"]["bsonType"] == "int"

    # Embedded array check
    assert "addresses" in props
    assert props["addresses"]["bsonType"] == "array"
    item_props = props["addresses"]["items"]["properties"]
    assert "street" in item_props
    assert "city" in item_props
    assert "customer_id" not in item_props  # Filtered out redundant FK


def test_stored_procedure_aggregation_transpilation():
    transpiler = StoredProcedureAggregationTranspiler()

    joins = [
        {"target_table": "order_items", "local_field": "order_id", "foreign_field": "order_id", "as": "items"}
    ]
    where_filters = {"status": "PAID", "region": "EMEA"}
    group_by = ["items.category"]
    aggregates = {
        "total_revenue": ("SUM", "items.subtotal"),
        "avg_discount": ("AVG", "items.discount"),
        "order_count": ("COUNT", "*"),
    }
    sort_by = {"total_revenue": -1}

    pipeline = transpiler.transpile_select_join_group(
        from_table="orders",
        joins=joins,
        where_filters=where_filters,
        group_by=group_by,
        aggregates=aggregates,
        sort_by=sort_by,
        limit=10,
    )

    assert len(pipeline) == 6
    assert "$match" in pipeline[0]
    assert pipeline[0]["$match"]["status"] == "PAID"
    assert "$lookup" in pipeline[1]
    assert pipeline[1]["$lookup"]["from"] == "order_items"
    assert "$unwind" in pipeline[2]
    assert pipeline[2]["$unwind"] == "$items"
    assert "$group" in pipeline[3]
    group_stage = pipeline[3]["$group"]
    assert group_stage["total_revenue"] == {"$sum": "$items.subtotal"}
    assert group_stage["avg_discount"] == {"$avg": "$items.discount"}
    assert group_stage["order_count"] == {"$sum": 1}
    assert "$sort" in pipeline[4]
    assert "$limit" in pipeline[5]
    assert pipeline[5]["$limit"] == 10


def test_analytics_facet_pipeline():
    transpiler = StoredProcedureAggregationTranspiler()
    match_criteria = {"tenant_id": "enterprise-corp-01"}
    facet_pipelines = {
        "status_summary": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}],
        "top_regions": [{"$group": {"_id": "$region", "revenue": {"$sum": "$amount"}}}, {"$sort": {"revenue": -1}}],
    }

    pipeline = transpiler.transpile_analytics_facet("transactions", match_criteria, facet_pipelines)
    assert len(pipeline) == 2
    assert "$match" in pipeline[0]
    assert "$facet" in pipeline[1]
    assert "status_summary" in pipeline[1]["$facet"]
    assert "top_regions" in pipeline[1]["$facet"]


def test_atlas_vector_search_and_semantic_cache():
    rag = AtlasVectorSearchRAGOrchestrator(cache_ttl_seconds=60)

    # Index sample migration docs
    rag.index_document(
        doc_id="doc_01",
        text="Transpiling Oracle PL/SQL cursors to MongoDB Aggregation Pipelines",
        embedding=[0.8, 0.6, 0.0],
        metadata={"tenant_id": "tenant-dublin", "category": "oracle_migration"},
    )
    rag.index_document(
        doc_id="doc_02",
        text="PostgreSQL foreign key constraints to MongoDB document referencing",
        embedding=[0.1, 0.9, 0.2],
        metadata={"tenant_id": "tenant-dublin", "category": "postgres_migration"},
    )
    rag.index_document(
        doc_id="doc_03",
        text="Enterprise multi-tenant data isolation on AWS EKS",
        embedding=[0.0, 0.1, 0.9],
        metadata={"tenant_id": "tenant-cork", "category": "security"},
    )

    # Search with filter
    results = rag.vector_search(
        query_embedding=[0.9, 0.5, 0.0],
        filter_metadata={"tenant_id": "tenant-dublin"},
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["id"] == "doc_01"
    assert results[0]["score"] > 0.85
    assert results[0]["metadata"]["category"] == "oracle_migration"

    # Search cached query
    cached_results = rag.vector_search(
        query_embedding=[0.9, 0.5, 0.0],
        filter_metadata={"tenant_id": "tenant-dublin"},
        top_k=2,
    )
    assert cached_results == results


def test_dual_query_consistency_validator():
    validator = DualQueryConsistencyValidator()

    relational_rows = [
        {"customer_id": "CUST_101", "total_spend": 1450.50, "orders_count": 5},
        {"customer_id": "CUST_102", "total_spend": 820.00, "orders_count": 3},
    ]
    mongo_results = [
        {"customer_id": "CUST_101", "total_spend": 1450.50, "orders_count": 5},
        {"customer_id": "CUST_102", "total_spend": 820.00, "orders_count": 3},
    ]

    report = validator.validate_consistency(
        relational_rows=relational_rows,
        document_results=mongo_results,
        key_field="customer_id",
        numeric_fields=["total_spend", "orders_count"],
    )

    assert report.is_consistent is True
    assert report.matched_keys == 2
    assert report.confidence_score == 1.0
    assert len(report.discrepancies) == 0

    # Test discrepancy detection
    divergent_mongo = [
        {"customer_id": "CUST_101", "total_spend": 1450.50, "orders_count": 5},
        {"customer_id": "CUST_102", "total_spend": 999.00, "orders_count": 3},  # mismatch
    ]
    report_div = validator.validate_consistency(
        relational_rows=relational_rows,
        document_results=divergent_mongo,
        key_field="customer_id",
        numeric_fields=["total_spend"],
    )
    assert report_div.is_consistent is False
    assert len(report_div.discrepancies) == 1
    assert "mismatch" in report_div.discrepancies[0]
