"""Unit and integration tests for Basecamp Research Platform Orchestrator & Bio-Agent Engine."""

import pytest

from llm_control_plane.basecamp_research_platform_orchestrator import (
    AssayType,
    BasecampResearchPlatformOrchestrator,
    BioAgentMCPServer,
    ComputeTarget,
    DistributedWorkflowDAG,
    GenomicDataIngestionPipeline,
    SequenceType,
    TaskStatus,
    WorkflowTask,
)

SAMPLE_FASTA = """
>BCR_GENE_001 Trillion Gene Atlas expedition sample Arctic
ATGCGATCGATCGATCGATCGATCGCGCGC
>BCR_GENE_002 Extremophile polymerase
ATGCATGCATGCATGCATGCATGC
"""

SAMPLE_FASTQ = """
@READ_001 Metagenomic ocean sequencing read
ATGCGATCGATC
+
IIIIIIIIIIII
@READ_002 Deep sea hydrothermal vent
CGCGCGCGCGCG
+
555555555555
"""


def test_fasta_parsing_and_metrics():
    pipeline = GenomicDataIngestionPipeline(min_phred_threshold=20.0)
    records = pipeline.parse_fasta(SAMPLE_FASTA, sequence_type=SequenceType.DNA)

    assert len(records) == 2
    rec1 = records[0]
    assert rec1.sequence_id == "BCR_GENE_001"
    assert rec1.length == 30
    assert rec1.gc_content > 50.0
    assert len(rec1.sha256_checksum) == 64
    assert rec1.is_valid is True
    assert len(rec1.qc_warnings) == 0


def test_invalid_characters_and_ambiguous_bases():
    pipeline = GenomicDataIngestionPipeline(max_n_ratio=0.10)
    bad_fasta = ">BAD_SEQ\nATGCZXNATGC\n"
    records = pipeline.parse_fasta(bad_fasta, sequence_type=SequenceType.DNA)

    assert len(records) == 1
    rec = records[0]
    assert rec.is_valid is False
    assert any("Invalid nucleotide characters detected" in w for w in rec.qc_warnings)


def test_fastq_parsing_and_phred_scores():
    pipeline = GenomicDataIngestionPipeline(min_phred_threshold=25.0)
    records = pipeline.parse_fastq(SAMPLE_FASTQ)

    assert len(records) == 2
    rec1 = records[0]
    assert rec1.sequence_id == "READ_001"
    assert rec1.length == 12
    # 'I' in Phred-33 is ASCII 73 - 33 = 40
    assert rec1.mean_phred_score == 40.0
    assert len(rec1.qc_warnings) == 0

    rec2 = records[1]
    # '5' in Phred-33 is ASCII 53 - 33 = 20 < 25 threshold
    assert rec2.mean_phred_score == 20.0
    assert any("below quality threshold" in w for w in rec2.qc_warnings)


def test_fastq_malformed_handling():
    pipeline = GenomicDataIngestionPipeline()
    malformed = "@SEQ_1\nATGC\n+\n"  # Only 3 lines
    with pytest.raises(ValueError, match="Line count"):
        pipeline.parse_fastq(malformed)


def test_workflow_dag_topological_sort_and_execution():
    dag = DistributedWorkflowDAG("test-wf-001")

    t1 = WorkflowTask("task-1", "Ingest", ComputeTarget.AWS_BATCH_CPU, "ingest.sh")
    t2 = WorkflowTask("task-2", "Quality Control", ComputeTarget.LOCAL_CONTAINER, "qc.sh", dependencies=["task-1"])
    t3 = WorkflowTask(
        "task-3",
        "EDEN Foundation Embedding",
        ComputeTarget.KUBERNETES_GPU_CLUSTER,
        "eden.sh",
        dependencies=["task-2"]
    )

    dag.add_task(t1)
    dag.add_task(t2)
    dag.add_task(t3)

    order = dag.validate_dag()
    assert order == ["task-1", "task-2", "task-3"]

    summary = dag.execute_all()
    assert summary["success"] is True
    assert summary["task_count"] == 3
    assert len(summary["lineage_hash"]) == 64
    assert dag.tasks["task-3"].status == TaskStatus.COMPLETED


def test_workflow_dag_cycle_detection():
    dag = DistributedWorkflowDAG("cycle-wf")
    t1 = WorkflowTask("task-A", "Task A", ComputeTarget.LOCAL_CONTAINER, "run.sh", dependencies=["task-B"])
    t2 = WorkflowTask("task-B", "Task B", ComputeTarget.LOCAL_CONTAINER, "run.sh", dependencies=["task-A"])

    dag.add_task(t1)
    dag.add_task(t2)

    with pytest.raises(ValueError, match="Cycle detected"):
        dag.validate_dag()


def test_workflow_dag_dependency_failure_handling():
    dag = DistributedWorkflowDAG("fail-wf")
    t1 = WorkflowTask("task-root", "Root Task", ComputeTarget.LOCAL_CONTAINER, "run.sh")
    t2 = WorkflowTask("task-child", "Child Task", ComputeTarget.LOCAL_CONTAINER, "child.sh", dependencies=["task-root"])

    dag.add_task(t1)
    dag.add_task(t2)

    def failing_executor(task: WorkflowTask):
        if task.task_id == "task-root":
            raise RuntimeError("Database connection timeout")
        return {"result": "OK"}

    summary = dag.execute_all(executor_fn=failing_executor)
    assert summary["success"] is False
    assert dag.tasks["task-child"].status == TaskStatus.FAILED
    assert "Dependency 'task-root' not completed" in (dag.tasks["task-child"].error or "")


def test_mcp_server_tools_and_invocations():
    pipeline = GenomicDataIngestionPipeline()
    mcp = BioAgentMCPServer(pipeline)

    tools = mcp.list_tools()
    tool_names = [t["name"] for t in tools]
    assert "validate_sequence" in tool_names
    assert "search_trillion_gene_atlas" in tool_names
    assert "dispatch_foundation_assay" in tool_names

    # Test tool invocation: validate_sequence
    res_val = mcp.call_tool("validate_sequence", {
        "format": "fasta",
        "content": ">SEQ1\nATGCGATCGATC\n",
        "sequence_type": "DNA"
    })
    assert res_val["status"] == "success"
    assert res_val["result"]["record_count"] == 1
    assert "latency_ms" in res_val["telemetry"]

    # Test tool invocation: search_trillion_gene_atlas
    res_search = mcp.call_tool("search_trillion_gene_atlas", {
        "kmer_motif": "ATGCGATC",
        "biome": "Hydrothermal Vent"
    })
    assert res_search["status"] == "success"
    assert res_search["result"]["matches_found"] > 0

    # Test tool invocation: dispatch_foundation_assay
    res_assay = mcp.call_tool("dispatch_foundation_assay", {
        "sequence_id": "BCR-001",
        "assay_type": AssayType.EDEN_FOUNDATION_EMBEDDING.value,
        "compute_target": ComputeTarget.KUBERNETES_GPU_CLUSTER.value
    })
    assert res_assay["status"] == "success"
    assert res_assay["result"]["status"] == "QUEUED"

    # Test unknown tool
    res_unk = mcp.call_tool("nonexistent_tool", {})
    assert res_unk["status"] == "error"


def test_end_to_end_platform_orchestration():
    orchestrator = BasecampResearchPlatformOrchestrator()
    result = orchestrator.ingest_and_benchmark_sample(
        fasta_content=SAMPLE_FASTA,
        sample_origin="Trillion Gene Atlas Patagonia"
    )

    assert result["total_ingested"] == 2
    assert result["valid_records"] == 2
    assert result["workflow_execution"]["success"] is True
    assert result["mcp_tools_available"] >= 3
