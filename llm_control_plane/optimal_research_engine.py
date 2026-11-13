"""Optimal Workshop Research Intelligence & Thematic Synthesis Engine.

Empowers UX researchers and product teams (Treejack, OptimalSort, Reframer)
with hierarchical observation clustering, participant agreement matrices,
information architecture navigation directness scoring, and AI-assisted qualitative
thematic synthesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class StudyType(str, Enum):
    CARD_SORT = "card_sort"
    TREE_TEST = "tree_test"
    QUALITATIVE_REFRAMER = "qualitative_reframer"


class StudyStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass
class ParticipantSession:
    session_id: str
    participant_id: str
    started_at: str
    completed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CardSortPlacement:
    participant_id: str
    card_id: str
    card_label: str
    category_id: str
    category_label: str


@dataclass
class TreeTestTaskResult:
    participant_id: str
    task_id: str
    selected_node_id: str
    path_taken: List[str]  # list of node_ids visited in order
    time_spent_seconds: float
    success: bool
    direct: bool


@dataclass
class QualitativeObservation:
    observation_id: str
    session_id: str
    raw_text: str
    tags: List[str] = field(default_factory=list)
    participant_quote: Optional[str] = None
    sentiment: str = "neutral"  # positive, neutral, negative
    impact_weight: float = 1.0


class ResearchObservationClusterer:
    """Computes co-occurrence matrices, pairwise similarity, and hierarchical clustering

    for card sorting studies (OptimalSort) and qualitative observations.
    """

    @staticmethod
    def calculate_cooccurrence_matrix(
        cards: List[str], placements: List[CardSortPlacement]
    ) -> Dict[str, Dict[str, int]]:
        """Builds symmetric co-occurrence frequency matrix for cards placed in same category."""
        matrix: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in cards} for c in cards}

        # Group cards by (participant_id, category_id)
        groupings: Dict[Tuple[str, str], List[str]] = {}
        for p in placements:
            if p.card_id not in matrix:
                continue
            key = (p.participant_id, p.category_id)
            groupings.setdefault(key, []).append(p.card_id)

        for card_list in groupings.values():
            unique_cards = list(set(card_list))
            for i in range(len(unique_cards)):
                c1 = unique_cards[i]
                for j in range(len(unique_cards)):
                    c2 = unique_cards[j]
                    matrix[c1][c2] += 1

        return matrix

    @staticmethod
    def calculate_similarity_matrix(
        cards: List[str],
        cooccurrence: Dict[str, Dict[str, int]],
        total_participants: int,
    ) -> Dict[str, Dict[str, float]]:
        """Calculates Jaccard / normalized co-occurrence agreement ratios (0.0 to 1.0)."""
        sim_matrix: Dict[str, Dict[str, float]] = {
            c: {c2: 0.0 for c2 in cards} for c in cards
        }
        if total_participants <= 0:
            return sim_matrix

        for c1 in cards:
            for c2 in cards:
                if c1 == c2:
                    sim_matrix[c1][c2] = 1.0
                else:
                    shared = cooccurrence.get(c1, {}).get(c2, 0)
                    sim_matrix[c1][c2] = round(shared / total_participants, 4)
        return sim_matrix

    @staticmethod
    def build_hierarchical_clusters(
        items: List[str],
        similarity_matrix: Dict[str, Dict[str, float]],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Agglomerative single-linkage clustering based on similarity threshold."""
        if not items:
            return []

        # Start each item as its own cluster
        clusters: List[Set[str]] = [{item} for item in items]

        # Iteratively merge clusters with max similarity >= threshold
        while len(clusters) > 1:
            best_sim = -1.0
            best_pair = (-1, -1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute average linkage similarity between clusters
                    sims = [
                        similarity_matrix.get(c1, {}).get(c2, 0.0)
                        for c1 in clusters[i]
                        for c2 in clusters[j]
                    ]
                    avg_sim = sum(sims) / len(sims) if sims else 0.0
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (i, j)

            if best_sim >= threshold and best_pair[0] != -1:
                i, j = best_pair
                merged = clusters[i].union(clusters[j])
                clusters.pop(j)
                clusters.pop(i)
                clusters.append(merged)
            else:
                break

        result = []
        for idx, cluster in enumerate(clusters, start=1):
            cluster_list = sorted(list(cluster))
            result.append(
                {
                    "cluster_id": f"cluster_{idx}",
                    "size": len(cluster_list),
                    "items": cluster_list,
                }
            )
        return result


class TreejackNavigationScorer:
    """Calculates navigation metrics, success rates, and directness for Treejack tree tests."""

    @staticmethod
    def evaluate_task(
        task_id: str,
        target_node_id: str,
        results: List[TreeTestTaskResult],
    ) -> Dict[str, Any]:
        """Calculates success, directness, and task efficiency across participants."""
        task_results = [r for r in results if r.task_id == task_id]
        total = len(task_results)
        if total == 0:
            return {
                "task_id": task_id,
                "participants_count": 0,
                "success_rate": 0.0,
                "directness_rate": 0.0,
                "direct_success_count": 0,
                "indirect_success_count": 0,
                "failure_count": 0,
                "avg_duration_seconds": 0.0,
                "efficiency_score": 0.0,
            }

        direct_success = 0
        indirect_success = 0
        direct_failure = 0
        indirect_failure = 0
        durations = []

        for r in task_results:
            is_success = r.selected_node_id == target_node_id
            durations.append(r.time_spent_seconds)
            if is_success:
                if r.direct:
                    direct_success += 1
                else:
                    indirect_success += 1
            else:
                if r.direct:
                    direct_failure += 1
                else:
                    indirect_failure += 1

        success_count = direct_success + indirect_success
        success_rate = round(success_count / total, 4)
        direct_count = direct_success + direct_failure
        directness_rate = round(direct_count / total, 4)
        avg_dur = round(sum(durations) / len(durations), 2)

        # Efficiency: harmonic-like balance of success and directness normalized by duration
        dur_penalty = min(1.0, 30.0 / max(1.0, avg_dur))
        efficiency_score = round(
            (0.6 * success_rate + 0.4 * directness_rate) * dur_penalty, 4
        )

        return {
            "task_id": task_id,
            "participants_count": total,
            "success_rate": success_rate,
            "directness_rate": directness_rate,
            "direct_success_count": direct_success,
            "indirect_success_count": indirect_success,
            "direct_failure_count": direct_failure,
            "indirect_failure_count": indirect_failure,
            "failure_count": total - success_count,
            "avg_duration_seconds": avg_dur,
            "efficiency_score": efficiency_score,
        }

    @staticmethod
    def identify_first_click_paths(
        results: List[TreeTestTaskResult],
    ) -> Dict[str, int]:
        """Distribution of participants' first clicks in the tree hierarchy."""
        first_clicks: Dict[str, int] = {}
        for r in results:
            if r.path_taken:
                root = r.path_taken[0]
                first_clicks[root] = first_clicks.get(root, 0) + 1
        return first_clicks


class ThematicSynthesisEngine:
    """Synthesizes qualitative observation notes (Reframer) into cohesive UX themes."""

    @staticmethod
    def extract_thematic_clusters(
        observations: List[QualitativeObservation],
        min_theme_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """Groups observations by shared tags and sentiments, calculating thematic significance."""
        if not observations:
            return []

        tag_buckets: Dict[str, List[QualitativeObservation]] = {}
        for obs in observations:
            if not obs.tags:
                tag_buckets.setdefault("unclassified", []).append(obs)
            else:
                for tag in obs.tags:
                    clean_tag = tag.strip().lower()
                    tag_buckets.setdefault(clean_tag, []).append(obs)

        themes: List[Dict[str, Any]] = []
        for tag, obs_list in tag_buckets.items():
            if len(obs_list) < min_theme_size:
                continue

            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            total_impact = 0.0
            evidence_quotes: List[str] = []

            for o in obs_list:
                s = o.sentiment.lower()
                if s in sentiment_counts:
                    sentiment_counts[s] += 1
                else:
                    sentiment_counts["neutral"] += 1
                total_impact += o.impact_weight
                if o.participant_quote:
                    evidence_quotes.append(o.participant_quote)

            total_obs = len(obs_list)
            neg_ratio = sentiment_counts["negative"] / total_obs
            # Opportunity score: weighted by volume, negative pain prevalence, and impact
            opp_score = round(
                (total_impact * 1.5) * (1.0 + neg_ratio) / math.sqrt(max(1, total_obs)),
                2,
            )

            themes.append(
                {
                    "theme_name": tag.capitalize(),
                    "observation_count": total_obs,
                    "sentiment_distribution": sentiment_counts,
                    "opportunity_score": opp_score,
                    "evidence_quotes": evidence_quotes[:5],
                    "recommendation": (
                        "High priority friction area requiring UX restructuring"
                        if neg_ratio >= 0.5
                        else "Stable user journey pattern with potential for optimization"
                    ),
                }
            )

        themes.sort(key=lambda t: t["opportunity_score"], reverse=True)
        return themes


class UserResearchStudyManager:
    """Orchestrates study lifecycles, collects participant inputs, and produces analytics."""

    def __init__(self) -> None:
        self.studies: Dict[str, Dict[str, Any]] = {}
        self.card_sort_placements: Dict[str, List[CardSortPlacement]] = {}
        self.tree_test_results: Dict[str, List[TreeTestTaskResult]] = {}
        self.observations: Dict[str, List[QualitativeObservation]] = {}

    def create_study(
        self,
        study_id: str,
        name: str,
        study_type: StudyType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Registers a new research study."""
        study = {
            "study_id": study_id,
            "name": name,
            "study_type": study_type.value,
            "status": StudyStatus.ACTIVE.value,
            "metadata": metadata or {},
            "participants_count": 0,
        }
        self.studies[study_id] = study
        return study

    def add_card_sort_placement(
        self,
        study_id: str,
        placement: CardSortPlacement,
    ) -> None:
        """Appends participant card placement."""
        self.card_sort_placements.setdefault(study_id, []).append(placement)
        if study_id in self.studies:
            pts = {p.participant_id for p in self.card_sort_placements[study_id]}
            self.studies[study_id]["participants_count"] = len(pts)

    def add_tree_test_result(
        self,
        study_id: str,
        result: TreeTestTaskResult,
    ) -> None:
        """Appends participant tree test result."""
        self.tree_test_results.setdefault(study_id, []).append(result)
        if study_id in self.studies:
            pts = {r.participant_id for r in self.tree_test_results[study_id]}
            self.studies[study_id]["participants_count"] = len(pts)

    def add_observation(
        self,
        study_id: str,
        obs: QualitativeObservation,
    ) -> None:
        """Appends qualitative observation note."""
        self.observations.setdefault(study_id, []).append(obs)

    def analyze_card_sort(
        self,
        study_id: str,
        cards: List[str],
        cluster_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Computes cooccurrence, agreement matrix, and hierarchical clusters."""
        placements = self.card_sort_placements.get(study_id, [])
        participants = len({p.participant_id for p in placements})
        cooc = ResearchObservationClusterer.calculate_cooccurrence_matrix(
            cards, placements
        )
        sim = ResearchObservationClusterer.calculate_similarity_matrix(
            cards, cooc, participants
        )
        clusters = ResearchObservationClusterer.build_hierarchical_clusters(
            cards, sim, cluster_threshold
        )
        return {
            "study_id": study_id,
            "total_participants": participants,
            "cooccurrence_matrix": cooc,
            "similarity_matrix": sim,
            "clusters": clusters,
        }

    def analyze_tree_test(
        self,
        study_id: str,
        task_id: str,
        target_node_id: str,
    ) -> Dict[str, Any]:
        """Evaluates tree test task success and directness metrics."""
        results = self.tree_test_results.get(study_id, [])
        eval_metrics = TreejackNavigationScorer.evaluate_task(
            task_id, target_node_id, results
        )
        first_clicks = TreejackNavigationScorer.identify_first_click_paths(results)
        return {
            "study_id": study_id,
            "metrics": eval_metrics,
            "first_click_distribution": first_clicks,
        }

    def synthesize_reframer_observations(
        self,
        study_id: str,
        min_theme_size: int = 1,
    ) -> Dict[str, Any]:
        """Synthesizes qualitative observation notes into prioritized themes."""
        obs = self.observations.get(study_id, [])
        themes = ThematicSynthesisEngine.extract_thematic_clusters(
            obs, min_theme_size=min_theme_size
        )
        return {
            "study_id": study_id,
            "total_observations": len(obs),
            "themes": themes,
        }
