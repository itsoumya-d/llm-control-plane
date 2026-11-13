"""Unit and integration tests for Optimal Workshop research intelligence engine."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from llm_control_plane import (
    CardSortPlacement,
    QualitativeObservation,
    ResearchObservationClusterer,
    StudyStatus,
    StudyType,
    ThematicSynthesisEngine,
    TreejackNavigationScorer,
    TreeTestTaskResult,
    UserResearchStudyManager,
)
from llm_control_plane.api import create_app


def test_card_sort_cooccurrence_and_similarity():
    cards = ["card_home", "card_pricing", "card_about", "card_contact"]
    placements = [
        CardSortPlacement("p1", "card_home", "Home", "cat_nav", "Navigation"),
        CardSortPlacement("p1", "card_about", "About Us", "cat_nav", "Navigation"),
        CardSortPlacement("p1", "card_pricing", "Pricing", "cat_sales", "Sales"),
        CardSortPlacement("p2", "card_home", "Home", "cat_main", "Main"),
        CardSortPlacement("p2", "card_about", "About Us", "cat_main", "Main"),
        CardSortPlacement("p2", "card_contact", "Contact", "cat_main", "Main"),
        CardSortPlacement("p3", "card_pricing", "Pricing", "cat_biz", "Business"),
        CardSortPlacement("p3", "card_contact", "Contact", "cat_biz", "Business"),
    ]

    cooc = ResearchObservationClusterer.calculate_cooccurrence_matrix(cards, placements)
    assert cooc["card_home"]["card_about"] == 2
    assert cooc["card_home"]["card_home"] == 2
    assert cooc["card_about"]["card_home"] == 2
    assert cooc["card_home"]["card_pricing"] == 0

    sim = ResearchObservationClusterer.calculate_similarity_matrix(cards, cooc, total_participants=3)
    assert sim["card_home"]["card_home"] == 1.0
    assert sim["card_home"]["card_about"] == pytest.approx(0.6667, abs=1e-3)
    assert sim["card_home"]["card_pricing"] == 0.0

    # Test empty / 0 participants
    empty_sim = ResearchObservationClusterer.calculate_similarity_matrix(cards, cooc, total_participants=0)
    assert empty_sim["card_home"]["card_about"] == 0.0


def test_hierarchical_clustering():
    cards = ["c1", "c2", "c3", "c4"]
    # c1 and c2 are tightly paired, c3 and c4 are loosely paired
    sim_matrix = {
        "c1": {"c1": 1.0, "c2": 0.9, "c3": 0.1, "c4": 0.1},
        "c2": {"c1": 0.9, "c2": 1.0, "c3": 0.1, "c4": 0.1},
        "c3": {"c1": 0.1, "c2": 0.1, "c3": 1.0, "c4": 0.6},
        "c4": {"c1": 0.1, "c2": 0.1, "c3": 0.6, "c4": 1.0},
    }

    clusters = ResearchObservationClusterer.build_hierarchical_clusters(cards, sim_matrix, threshold=0.5)
    assert len(clusters) == 2
    cluster_sets = [set(c["items"]) for c in clusters]
    assert {"c1", "c2"} in cluster_sets
    assert {"c3", "c4"} in cluster_sets

    # Empty items
    assert ResearchObservationClusterer.build_hierarchical_clusters([], sim_matrix) == []


def test_treejack_navigation_scorer():
    results = [
        TreeTestTaskResult("p1", "task_find_pricing", "node_pricing", ["root", "node_pricing"], 12.5, True, True),
        TreeTestTaskResult(
            "p2", "task_find_pricing", "node_pricing", ["root", "node_about", "node_pricing"], 25.0, True, False
        ),
        TreeTestTaskResult("p3", "task_find_pricing", "node_faq", ["root", "node_faq"], 15.0, False, True),
        TreeTestTaskResult(
            "p4", "task_find_pricing", "node_support", ["root", "node_faq", "node_support"], 45.0, False, False
        ),
    ]

    metrics = TreejackNavigationScorer.evaluate_task("task_find_pricing", "node_pricing", results)
    assert metrics["participants_count"] == 4
    assert metrics["success_rate"] == 0.5  # 2 out of 4
    assert metrics["directness_rate"] == 0.5  # 2 out of 4 (p1 direct success, p3 direct failure)
    assert metrics["direct_success_count"] == 1
    assert metrics["indirect_success_count"] == 1
    assert metrics["direct_failure_count"] == 1
    assert metrics["indirect_failure_count"] == 1
    assert metrics["failure_count"] == 2
    assert metrics["avg_duration_seconds"] == pytest.approx(24.38, abs=0.1)
    assert metrics["efficiency_score"] > 0.0

    # Test empty task results
    empty_eval = TreejackNavigationScorer.evaluate_task("nonexistent", "node_pricing", results)
    assert empty_eval["participants_count"] == 0
    assert empty_eval["success_rate"] == 0.0

    first_clicks = TreejackNavigationScorer.identify_first_click_paths(results)
    assert first_clicks["root"] == 4


def test_thematic_synthesis_engine():
    observations = [
        QualitativeObservation(
            "obs_1",
            "session_1",
            "Checkout button was obscured by sticky footer",
            tags=["checkout", "mobile"],
            participant_quote="I could not find where to pay on my phone",
            sentiment="negative",
            impact_weight=2.0,
        ),
        QualitativeObservation(
            "obs_2",
            "session_2",
            "Checkout process felt seamless on desktop",
            tags=["checkout", "desktop"],
            participant_quote="One click and I was done",
            sentiment="positive",
            impact_weight=1.0,
        ),
        QualitativeObservation(
            "obs_3",
            "session_3",
            "Search bar returned zero results for typo",
            tags=["search"],
            sentiment="negative",
            impact_weight=1.5,
        ),
        QualitativeObservation(
            "obs_4",
            "session_4",
            "General feedback note with no tags",
            tags=[],
            sentiment="neutral",
            impact_weight=0.5,
        ),
    ]

    themes = ThematicSynthesisEngine.extract_thematic_clusters(observations, min_theme_size=1)
    assert len(themes) == 5  # checkout, mobile, desktop, search, unclassified
    checkout_theme = next(t for t in themes if t["theme_name"] == "Checkout")
    assert checkout_theme["observation_count"] == 2
    assert checkout_theme["sentiment_distribution"]["negative"] == 1
    assert checkout_theme["sentiment_distribution"]["positive"] == 1
    assert len(checkout_theme["evidence_quotes"]) == 2

    # High negative friction recommendation check
    search_theme = next(t for t in themes if t["theme_name"] == "Search")
    assert "High priority" in search_theme["recommendation"]

    # Test empty observations
    assert ThematicSynthesisEngine.extract_thematic_clusters([]) == []


def test_user_research_study_manager_lifecycle():
    mgr = UserResearchStudyManager()
    study = mgr.create_study("study_ia_v1", "E-commerce IA Benchmark", StudyType.CARD_SORT, {"domain": "retail"})
    assert study["study_id"] == "study_ia_v1"
    assert study["status"] == StudyStatus.ACTIVE.value

    # Card sort flow
    mgr.add_card_sort_placement(
        "study_ia_v1", CardSortPlacement("p1", "card_cart", "Cart", "cat_checkout", "Checkout")
    )
    mgr.add_card_sort_placement(
        "study_ia_v1", CardSortPlacement("p1", "card_pay", "Payment", "cat_checkout", "Checkout")
    )
    analysis = mgr.analyze_card_sort("study_ia_v1", ["card_cart", "card_pay"])
    assert analysis["total_participants"] == 1
    assert len(analysis["clusters"]) == 1

    # Tree test flow
    mgr.add_tree_test_result(
        "study_ia_v1",
        TreeTestTaskResult("p1", "t1", "node_target", ["home", "node_target"], 8.0, True, True),
    )
    tree_eval = mgr.analyze_tree_test("study_ia_v1", "t1", "node_target")
    assert tree_eval["metrics"]["success_rate"] == 1.0

    # Reframer observations flow
    mgr.add_observation(
        "study_ia_v1",
        QualitativeObservation("obs_10", "sess_10", "User struggled with dropdown", ["dropdown"], sentiment="negative"),
    )
    synth = mgr.synthesize_reframer_observations("study_ia_v1")
    assert synth["total_observations"] == 1
    assert synth["themes"][0]["theme_name"] == "Dropdown"


def test_fastapi_research_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Create study
    res = client.post(
        "/v1/research/studies",
        json={
            "study_id": "study_optimal_01",
            "name": "Navigation Structure Testing",
            "study_type": "tree_test",
            "metadata": {"platform": "Treejack"},
        },
    )
    assert res.status_code == 200
    assert res.json()["study"]["study_id"] == "study_optimal_01"

    # Invalid study type error
    bad_res = client.post(
        "/v1/research/studies",
        json={"study_id": "bad", "name": "bad", "study_type": "invalid_type"},
    )
    assert bad_res.status_code == 400
    assert bad_res.json()["error"] == "invalid_study_type"

    # 2. Analyze card sort
    card_res = client.post(
        "/v1/research/cardsort/analyze",
        json={
            "study_id": "cs_01",
            "cards": ["card_a", "card_b"],
            "placements": [
                {
                    "participant_id": "p1",
                    "card_id": "card_a",
                    "card_label": "A",
                    "category_id": "cat_1",
                    "category_label": "Category 1",
                },
                {
                    "participant_id": "p1",
                    "card_id": "card_b",
                    "card_label": "B",
                    "category_id": "cat_1",
                    "category_label": "Category 1",
                },
            ],
            "cluster_threshold": 0.5,
        },
    )
    assert card_res.status_code == 200
    data = card_res.json()
    assert data["total_participants"] == 1
    assert data["similarity_matrix"]["card_a"]["card_b"] == 1.0

    # 3. Evaluate treejack
    tree_res = client.post(
        "/v1/research/treejack/evaluate",
        json={
            "study_id": "tj_01",
            "task_id": "task_1",
            "target_node_id": "dest",
            "results": [
                {
                    "participant_id": "p1",
                    "task_id": "task_1",
                    "selected_node_id": "dest",
                    "path_taken": ["home", "dest"],
                    "time_spent_seconds": 10.0,
                    "success": True,
                    "direct": True,
                }
            ],
        },
    )
    assert tree_res.status_code == 200
    assert tree_res.json()["metrics"]["success_rate"] == 1.0

    # 4. Synthesize reframer
    reframer_res = client.post(
        "/v1/research/reframer/synthesize",
        json={
            "study_id": "ref_01",
            "observations": [
                {
                    "observation_id": "o1",
                    "session_id": "s1",
                    "raw_text": "Participant praised clear typography",
                    "tags": ["typography"],
                    "participant_quote": "Very readable",
                    "sentiment": "positive",
                    "impact_weight": 1.2,
                }
            ],
            "min_theme_size": 1,
        },
    )
    assert reframer_res.status_code == 200
    assert len(reframer_res.json()["themes"]) == 1
    assert reframer_res.json()["themes"][0]["theme_name"] == "Typography"
