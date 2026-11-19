"""Unit tests for market_ai_engine (Alternative Data, PyTorch Fine-Tuning, Model Evaluation)."""

import pytest

from llm_control_plane.market_ai_engine import (
    AlternativeDataPreprocessor,
    DeepLearningFineTuner,
    MarketAIService,
    MarketNLPFeatureExtractor,
    MarketRecord,
    ModelEvaluationBenchmark,
)


def test_alternative_data_preprocessor_clean():
    prep = AlternativeDataPreprocessor()
    raw = "   AMAZON.COM*RETAIL   $129.99   \n\t  ORDER #9982   "
    cleaned = prep.clean_text(raw)
    assert "amazon.com*retail" in cleaned
    assert "\t" not in cleaned


def test_alternative_data_preprocessor_entities():
    prep = AlternativeDataPreprocessor()
    raw = "Stripe Payment invoice €450.50 processed successfully"
    res = prep.extract_entities(raw)
    assert res["merchant"] == "stripe"
    assert 450.50 in res["detected_amounts"]
    assert res["token_count"] == 6


def test_alternative_data_preprocessor_records():
    prep = AlternativeDataPreprocessor()
    data = [
        {"id": "r1", "raw_text": "Netflix streaming sub $19.99", "category": "entertainment"},
        {"id": "r2", "raw_text": "AWS Cloud Services ₹8500.00", "category": "cloud"}
    ]
    records = prep.process_records(data)
    assert len(records) == 2
    assert isinstance(records[0], MarketRecord)
    assert records[0].merchant == "netflix"
    assert records[0].amount == 19.99


def test_market_nlp_feature_extractor():
    extractor = MarketNLPFeatureExtractor(vocab_size=100)
    corpus = [
        "AWS Cloud computing infrastructure",
        "E-commerce alternative consumer spending telemetry",
        "Market analytics predictive intelligence"
    ]
    extractor.fit(corpus)
    assert extractor._is_fitted
    assert len(extractor.vocab) > 5

    seq = extractor.transform("AWS cloud telemetry", max_seq_len=8)
    assert len(seq) == 8
    assert seq[0] != 0  # Not pad


def test_feature_extractor_unfitted_raises():
    extractor = MarketNLPFeatureExtractor()
    with pytest.raises(RuntimeError):
        extractor.transform("test")


def test_deep_learning_fine_tuner():
    tuner = DeepLearningFineTuner(input_dim=8, hidden_dim=4, num_classes=2)
    sample_x = [0.1 * i for i in range(8)]
    probs = tuner.forward(sample_x)
    assert len(probs) == 2
    assert pytest.approx(sum(probs), 0.001) == 1.0


def test_deep_learning_fine_tuner_training_loop():
    tuner = DeepLearningFineTuner(input_dim=8, hidden_dim=4, num_classes=2)
    batch_x = [[0.2 * i for i in range(8)] for _ in range(5)]
    batch_y = [1, 0, 1, 1, 0]
    res = tuner.fine_tune(batch_x, batch_y, epochs=3)
    assert res["status"] == "completed"
    assert res["epochs"] == 3
    assert len(res["history"]) == 3
    assert res["final_loss"] > 0.0


def test_model_evaluation_benchmark_metrics():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
    metrics = ModelEvaluationBenchmark.compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_model_evaluation_benchmark_empty():
    metrics = ModelEvaluationBenchmark.compute_classification_metrics([], [])
    assert metrics["accuracy"] == 0.0


def test_model_evaluation_benchmark_perplexity():
    losses = [0.2, 0.4, 0.3]
    ppl = ModelEvaluationBenchmark.compute_perplexity(losses)
    assert ppl > 1.0


def test_model_evaluation_benchmark_latency():
    def dummy_fn(x):
        return [v * 2 for v in x]
    
    samples = [[1.0, 2.0] for _ in range(10)]
    latency = ModelEvaluationBenchmark.evaluate_inference_latency(dummy_fn, samples, runs=5)
    assert "p50_ms" in latency
    assert "p95_ms" in latency
    assert latency["p50_ms"] >= 0.0


def test_market_ai_service_predict_and_eval():
    svc = MarketAIService()
    res = svc.predict("Apple App Store subscription $14.99")
    assert "prediction" in res
    assert "probabilities" in res
    assert len(res["probabilities"]) == 2

    batch = [
        ("Amazon retail invoice $45.00", 1),
        ("Unknown noise packet error 404", 0),
        ("Stripe online payment $120.00", 1)
    ]
    eval_res = svc.evaluate_batch(batch)
    assert eval_res["sample_count"] == 3
    assert "accuracy" in eval_res["metrics"]
    assert eval_res["perplexity"] >= 1.0
