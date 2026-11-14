"""MongoDB Forward Deployed AI Modernizer & RAG Orchestration Engine.

Engineered for MongoDB enterprise application modernization initiatives:
- Relational Schema Analyzer: AST parser for SQL DDL (PostgreSQL, Oracle, MySQL, SQL Server)
  analyzing relationships, cardinalities, and foreign keys.
- Document Schema Synthesizer: Heuristic & AI-driven document modeling engine generating
  MongoDB JSON Schema validation ($jsonSchema) with intelligent embedding vs referencing decisions.
- Stored Procedure Transpiler: Transpilation engine compiling relational queries and stored
  procedure patterns into idiomatic MongoDB Aggregation Pipelines ($match, $lookup, $unwind, $group, $facet).
- Atlas Vector Search & Semantic Cache: Multi-tenant vector retrieval engine with metadata
  filtering, dimension normalization, and high-performance TTL semantic caching.
- Dual-Query Consistency Validator: Automated correctness verification engine benchmarking
  legacy relational query result sets against MongoDB document aggregation outputs.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class DocumentModelingStrategy(str, enum.Enum):
    """Document modeling strategy for relational tables."""

    EMBED = "EMBED"
    REFERENCE = "REFERENCE"
    HYBRID = "HYBRID"


class SQLDataType(str, enum.Enum):
    """Supported SQL data types mapped to BSON / JSON Schema."""

    INT = "INT"
    BIGINT = "BIGINT"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    DECIMAL = "DECIMAL"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "TIMESTAMP"
    JSON = "JSON"


@dataclass
class SQLColumn:
    """Relational table column definition."""

    name: str
    data_type: SQLDataType
    is_primary_key: bool = False
    is_nullable: bool = True
    default_value: Optional[str] = None


@dataclass
class SQLForeignKey:
    """Foreign key relationship definition."""

    column_name: str
    target_table: str
    target_column: str
    is_unique: bool = False  # Unique FK implies 1:1, otherwise 1:N
    is_unbounded: bool = False  # High write-rate / unbounded collection


@dataclass
class SQLTable:
    """Relational table schema."""

    name: str
    columns: List[SQLColumn]
    foreign_keys: List[SQLForeignKey] = field(default_factory=list)
    estimated_cardinality: int = 1000


@dataclass
class ValidationReport:
    """Consistency validation report comparing relational vs document execution."""

    is_consistent: bool
    relational_row_count: int
    document_count: int
    matched_keys: int
    discrepancies: List[str]
    confidence_score: float


class RelationalSchemaAnalyzer:
    """Parses SQL DDL and derives optimal MongoDB document modeling strategies."""

    # SQL data type mappings
    TYPE_MAP = {
        "int": SQLDataType.INT,
        "integer": SQLDataType.INT,
        "bigint": SQLDataType.BIGINT,
        "varchar": SQLDataType.VARCHAR,
        "text": SQLDataType.TEXT,
        "numeric": SQLDataType.DECIMAL,
        "decimal": SQLDataType.DECIMAL,
        "boolean": SQLDataType.BOOLEAN,
        "bool": SQLDataType.BOOLEAN,
        "timestamp": SQLDataType.TIMESTAMP,
        "timestamptz": SQLDataType.TIMESTAMP,
        "date": SQLDataType.TIMESTAMP,
        "json": SQLDataType.JSON,
        "jsonb": SQLDataType.JSON,
    }

    def parse_table_ddl(self, ddl: str) -> SQLTable:
        """Parses a CREATE TABLE SQL statement into an SQLTable object."""
        table_match = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_]+)", ddl, re.IGNORECASE)
        if not table_match:
            raise ValueError(f"Could not parse table name from DDL: {ddl}")
        table_name = table_match.group(1).lower()

        # Extract columns and constraints inside parentheses
        body_match = re.search(r"\((.*)\)", ddl, re.DOTALL)
        if not body_match:
            return SQLTable(name=table_name, columns=[])

        body = body_match.group(1)
        lines = [line.strip().rstrip(",") for line in body.split("\n") if line.strip()]

        columns: List[SQLColumn] = []
        foreign_keys: List[SQLForeignKey] = []
        pk_columns: Set[str] = set()

        # Find explicit PRIMARY KEY constraints
        for line in lines:
            pk_match = re.search(r"PRIMARY\s+KEY\s*\(([a-zA-Z0-9_,\s]+)\)", line, re.IGNORECASE)
            if pk_match:
                for c in pk_match.group(1).split(","):
                    pk_columns.add(c.strip().lower())

        for line in lines:
            line_clean = line.strip().rstrip(",")
            if not line_clean or line_clean.upper().startswith("CONSTRAINT") or line_clean.upper().startswith("PRIMARY KEY"):
                continue

            # Foreign key constraint check
            fk_match = re.search(
                r"FOREIGN\s+KEY\s*\(([a-zA-Z0-9_]+)\)\s*REFERENCES\s+([a-zA-Z0-9_]+)\s*\(([a-zA-Z0-9_]+)\)",
                line_clean,
                re.IGNORECASE,
            )
            if fk_match:
                foreign_keys.append(
                    SQLForeignKey(
                        column_name=fk_match.group(1).lower(),
                        target_table=fk_match.group(2).lower(),
                        target_column=fk_match.group(3).lower(),
                    )
                )
                continue

            parts = line_clean.split()
            if len(parts) >= 2:
                col_name = parts[0].strip('"`[]').lower()
                type_raw = parts[1].split("(")[0].lower()
                data_type = self.TYPE_MAP.get(type_raw, SQLDataType.VARCHAR)
                is_pk = "PRIMARY KEY" in line_clean.upper() or col_name in pk_columns
                is_nullable = "NOT NULL" not in line_clean.upper() and not is_pk

                columns.append(
                    SQLColumn(
                        name=col_name,
                        data_type=data_type,
                        is_primary_key=is_pk,
                        is_nullable=is_nullable,
                    )
                )

        return SQLTable(name=table_name, columns=columns, foreign_keys=foreign_keys)

    def determine_strategy(
        self, parent_table: SQLTable, child_table: SQLTable, fk: SQLForeignKey
    ) -> DocumentModelingStrategy:
        """Determines EMBED vs REFERENCE strategy based on MongoDB document modeling principles.

        - 1:1 or 1:bounded-N (<= 500 items, low write contention) -> EMBED
        - 1:unbounded-N or high-write volume -> REFERENCE (to avoid 16MB BSON limit and document growth penalty)
        """
        if fk.is_unbounded:
            return DocumentModelingStrategy.REFERENCE
        if fk.is_unique:
            return DocumentModelingStrategy.EMBED
        # If child estimated cardinality is low, default to EMBED; if high, REFERENCE
        if child_table.estimated_cardinality > 2000:
            return DocumentModelingStrategy.REFERENCE
        return DocumentModelingStrategy.EMBED


class MongoDBDocumentSchemaSynthesizer:
    """Synthesizes production MongoDB $jsonSchema validators and document models."""

    BSON_TYPE_MAP = {
        SQLDataType.INT: "int",
        SQLDataType.BIGINT: "long",
        SQLDataType.VARCHAR: "string",
        SQLDataType.TEXT: "string",
        SQLDataType.DECIMAL: "decimal",
        SQLDataType.BOOLEAN: "bool",
        SQLDataType.TIMESTAMP: "date",
        SQLDataType.JSON: "object",
    }

    def generate_json_schema(
        self,
        primary_table: SQLTable,
        embedded_tables: Optional[List[Tuple[SQLTable, SQLForeignKey]]] = None,
    ) -> Dict[str, Any]:
        """Generates a MongoDB $jsonSchema validator object."""
        properties: Dict[str, Any] = {
            "_id": {"bsonType": "objectId", "description": "Primary document ObjectId identifier"}
        }
        required_fields: List[str] = ["_id"]

        for col in primary_table.columns:
            bson_type = self.BSON_TYPE_MAP.get(col.data_type, "string")
            col_spec: Dict[str, Any] = {"bsonType": bson_type, "description": f"Field migrated from {col.name}"}
            properties[col.name] = col_spec
            if not col.is_nullable:
                required_fields.append(col.name)

        if embedded_tables:
            for child_table, fk in embedded_tables:
                child_props: Dict[str, Any] = {}
                child_reqs: List[str] = []
                for c in child_table.columns:
                    if c.name == fk.column_name:
                        continue  # Skip redundant foreign key in embedded array
                    b_type = self.BSON_TYPE_MAP.get(c.data_type, "string")
                    child_props[c.name] = {"bsonType": b_type}
                    if not c.is_nullable:
                        child_reqs.append(c.name)

                if fk.is_unique:
                    # 1:1 Embedded Subdocument
                    properties[child_table.name] = {
                        "bsonType": "object",
                        "required": child_reqs,
                        "properties": child_props,
                    }
                else:
                    # 1:N Embedded Array of Documents
                    properties[child_table.name] = {
                        "bsonType": "array",
                        "items": {
                            "bsonType": "object",
                            "required": child_reqs,
                            "properties": child_props,
                        },
                        "description": f"Embedded subdocuments from table {child_table.name}",
                    }

        return {
            "$jsonSchema": {
                "bsonType": "object",
                "required": required_fields,
                "properties": properties,
            }
        }


class StoredProcedureAggregationTranspiler:
    """Transpiles relational SQL statements & procedures into MongoDB Aggregation Pipelines."""

    def transpile_select_join_group(
        self,
        from_table: str,
        joins: List[Dict[str, str]],
        where_filters: Dict[str, Any],
        group_by: List[str],
        aggregates: Dict[str, Tuple[str, str]],  # alias -> (operator, column)
        sort_by: Optional[Dict[str, int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compiles a relational SQL query with JOINs, GROUP BY, and aggregations into a pipeline."""
        pipeline: List[Dict[str, Any]] = []

        # 1. $match stage for where_filters
        if where_filters:
            match_doc: Dict[str, Any] = {}
            for col, val in where_filters.items():
                if isinstance(val, dict):
                    match_doc[col] = val
                else:
                    match_doc[col] = val
            pipeline.append({"$match": match_doc})

        # 2. $lookup & $unwind stages for joins
        for j in joins:
            target_table = j["target_table"]
            local_field = j["local_field"]
            foreign_field = j["foreign_field"]
            as_field = j.get("as", target_table)

            pipeline.append(
                {
                    "$lookup": {
                        "from": target_table,
                        "localField": local_field,
                        "foreignField": foreign_field,
                        "as": as_field,
                    }
                }
            )
            # Unwind if inner or 1:1 join
            if j.get("unwind", True):
                pipeline.append({"$unwind": f"${as_field}"})

        # 3. $group stage
        group_id: Any = None
        if len(group_by) == 1:
            group_id = f"${group_by[0]}"
        elif len(group_by) > 1:
            group_id = {col.replace(".", "_"): f"${col}" for col in group_by}

        group_doc: Dict[str, Any] = {"_id": group_id}
        for alias, (op, col) in aggregates.items():
            op_upper = op.upper()
            if op_upper == "SUM":
                group_doc[alias] = {"$sum": f"${col}"}
            elif op_upper == "AVG":
                group_doc[alias] = {"$avg": f"${col}"}
            elif op_upper == "COUNT":
                group_doc[alias] = {"$sum": 1}
            elif op_upper == "MAX":
                group_doc[alias] = {"$max": f"${col}"}
            elif op_upper == "MIN":
                group_doc[alias] = {"$min": f"${col}"}
            else:
                group_doc[alias] = {"$first": f"${col}"}

        pipeline.append({"$group": group_doc})

        # 4. $sort stage
        if sort_by:
            pipeline.append({"$sort": sort_by})

        # 5. $limit stage
        if limit is not None:
            pipeline.append({"$limit": limit})

        return pipeline

    def transpile_analytics_facet(
        self,
        base_collection: str,
        match_criteria: Dict[str, Any],
        facet_pipelines: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Compiles multi-faceted analytics queries into MongoDB $facet pipelines."""
        pipeline: List[Dict[str, Any]] = []
        if match_criteria:
            pipeline.append({"$match": match_criteria})
        pipeline.append({"$facet": facet_pipelines})
        return pipeline


class AtlasVectorSearchRAGOrchestrator:
    """Manages Atlas Vector Search indexing, semantic similarity, and TTL caching."""

    def __init__(self, cache_ttl_seconds: int = 3600):
        self.cache_ttl = cache_ttl_seconds
        self._semantic_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._vector_store: List[Dict[str, Any]] = []

    def build_vector_index_spec(
        self,
        index_name: str,
        path: str,
        num_dimensions: int = 1536,
        similarity: str = "cosine",
        filter_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generates Atlas Vector Search Index definition (JSON for Atlas Admin API)."""
        fields_spec: List[Dict[str, Any]] = [
            {
                "type": "vector",
                "path": path,
                "numDimensions": num_dimensions,
                "similarity": similarity,
            }
        ]
        if filter_fields:
            for f in filter_fields:
                fields_spec.append({"type": "filter", "path": f})

        return {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {"fields": fields_spec},
        }

    def index_document(self, doc_id: str, text: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Stores a vector document into the local orchestrator store."""
        # Normalize embedding vector
        norm = math.sqrt(sum(x * x for x in embedding)) or 1.0
        normalized_emb = [x / norm for x in embedding]

        self._vector_store.append(
            {
                "id": doc_id,
                "text": text,
                "embedding": normalized_emb,
                "metadata": metadata,
            }
        )

    def vector_search(
        self,
        query_embedding: List[float],
        filter_metadata: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Executes vector search with cosine similarity and metadata pre-filtering."""
        # Compute query hash for cache lookup
        q_norm = math.sqrt(sum(x * x for x in query_embedding)) or 1.0
        norm_query = [x / q_norm for x in query_embedding]

        cache_key = hashlib.sha256(
            (json.dumps(norm_query[:10]) + json.dumps(filter_metadata or {}, sort_keys=True)).encode()
        ).hexdigest()

        now = time.time()
        if cache_key in self._semantic_cache:
            cached_time, results = self._semantic_cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return results

        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for doc in self._vector_store:
            # Metadata filter check
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if doc["metadata"].get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Dot product for cosine similarity on normalized vectors
            sim = sum(q * d for q, d in zip(norm_query, doc["embedding"]))
            if sim >= min_score:
                candidates.append((sim, doc))

        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [
            {
                "id": doc["id"],
                "text": doc["text"],
                "score": round(score, 4),
                "metadata": doc["metadata"],
            }
            for score, doc in candidates[:top_k]
        ]

        self._semantic_cache[cache_key] = (now, results)
        return results


class DualQueryConsistencyValidator:
    """Verifies semantic correctness between relational queries and MongoDB aggregation pipelines."""

    def validate_consistency(
        self,
        relational_rows: List[Dict[str, Any]],
        document_results: List[Dict[str, Any]],
        key_field: str,
        numeric_fields: Optional[List[str]] = None,
        tolerance: float = 1e-4,
    ) -> ValidationReport:
        """Validates that relational rows match MongoDB aggregation results."""
        discrepancies: List[str] = []
        rel_map = {str(r[key_field]): r for r in relational_rows if key_field in r}
        doc_map = {str(d[key_field]): d for d in document_results if key_field in d}

        matched_keys = 0
        all_keys = set(rel_map.keys()) | set(doc_map.keys())

        for k in all_keys:
            if k not in rel_map:
                discrepancies.append(f"Key {k} present in MongoDB results but missing in relational")
                continue
            if k not in doc_map:
                discrepancies.append(f"Key {k} present in relational results but missing in MongoDB")
                continue

            matched_keys += 1
            rel_row = rel_map[k]
            doc_row = doc_map[k]

            if numeric_fields:
                for num_col in numeric_fields:
                    r_val = float(rel_row.get(num_col, 0.0))
                    d_val = float(doc_row.get(num_col, 0.0))
                    if abs(r_val - d_val) > tolerance:
                        discrepancies.append(
                            f"Key {k} metric '{num_col}' mismatch: relational={r_val}, mongo={d_val}"
                        )

        is_consistent = len(discrepancies) == 0
        total_items = max(len(all_keys), 1)
        confidence = max(0.0, 1.0 - (len(discrepancies) / total_items))

        return ValidationReport(
            is_consistent=is_consistent,
            relational_row_count=len(relational_rows),
            document_count=len(document_results),
            matched_keys=matched_keys,
            discrepancies=discrepancies,
            confidence_score=round(confidence, 4),
        )
