"""Alternative market data intelligence, deep learning fine-tuning, and model evaluation engine.

Implements end-to-end NLP data preprocessing, deep learning model training harness,
multi-metric model evaluation benchmarking, and low-latency production API serving.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MarketRecord:
    record_id: str
    raw_text: str
    merchant: Optional[str] = None
    category: Optional[str] = None
    amount: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlternativeDataPreprocessor:
    """Preprocesses raw alternative market telemetry data (receipts, web scrapes, transactions)."""

    def __init__(self, lowercase: bool = True, strip_noise: bool = True):
        self.lowercase = lowercase
        self.strip_noise = strip_noise
        self._price_regex = re.compile(r"[\$€£₹]?\s*(\d+(?:[.,]\d{2})?)")
        self._merchant_cleaner = re.compile(r"[^a-zA-Z0-9\s]")

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        if self.lowercase:
            text = text.lower()
        if self.strip_noise:
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"[\r\n\t]+", " ", text)
        return text

    def extract_entities(self, text: str) -> Dict[str, Any]:
        cleaned = self.clean_text(text)
        amounts = []
        for m in self._price_regex.finditer(cleaned):
            try:
                amounts.append(float(m.group(1).replace(",", "")))
            except ValueError:
                pass
        
        # Simple merchant pattern extraction
        tokens = cleaned.split()
        merchant = tokens[0] if tokens else "unknown"
        merchant = self._merchant_cleaner.sub("", merchant)

        return {
            "cleaned_text": cleaned,
            "merchant": merchant,
            "detected_amounts": amounts,
            "token_count": len(tokens)
        }

    def process_records(self, records: List[Dict[str, Any]]) -> List[MarketRecord]:
        processed = []
        for r in records:
            raw = r.get("raw_text", "")
            entities = self.extract_entities(raw)
            rec = MarketRecord(
                record_id=r.get("id", str(time.time())),
                raw_text=raw,
                merchant=r.get("merchant") or entities["merchant"],
                category=r.get("category", "retail"),
                amount=r.get("amount") or (entities["detected_amounts"][0] if entities["detected_amounts"] else 0.0),
                metadata=entities
            )
            processed.append(rec)
        return processed


class MarketNLPFeatureExtractor:
    """Extracts dense feature representations and vocabulary encodings for NLP models."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self._is_fitted = False

    def fit(self, texts: List[str]) -> "MarketNLPFeatureExtractor":
        counts: Dict[str, int] = {}
        for t in texts:
            words = re.findall(r"\w+", t.lower())
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        
        sorted_words = sorted(counts.items(), key=lambda x: -x[1])[:self.vocab_size - 2]
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for idx, (w, _) in enumerate(sorted_words, start=2):
            self.vocab[w] = idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._is_fitted = True
        return self

    def transform(self, text: str, max_seq_len: int = 32) -> List[int]:
        if not self._is_fitted:
            raise RuntimeError("MarketNLPFeatureExtractor must be fitted before transform")
        words = re.findall(r"\w+", text.lower())
        indices = [self.vocab.get(w, 1) for w in words[:max_seq_len]]
        if len(indices) < max_seq_len:
            indices.extend([0] * (max_seq_len - len(indices)))
        return indices


class DeepLearningFineTuner:
    """Simulates a PyTorch-style gradient-based deep learning fine-tuning and training harness."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 16, num_classes: int = 2, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        # Initialize weights deterministically
        self.w1 = [[0.01 * (i + j) for j in range(hidden_dim)] for i in range(input_dim)]
        self.w2 = [[0.01 * (i + j) for j in range(num_classes)] for i in range(hidden_dim)]
        self.history: List[Dict[str, float]] = []

    def _softmax(self, logits: List[float]) -> List[float]:
        max_val = max(logits)
        exp_vals = [math.exp(v - max_val) for v in logits]
        s = sum(exp_vals)
        return [v / (s + 1e-9) for v in exp_vals]

    def forward(self, x: List[float]) -> List[float]:
        # Hidden layer with ReLU
        hidden = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            s = sum(x[i] * self.w1[i][j] for i in range(min(len(x), self.input_dim)))
            hidden[j] = max(0.0, s)

        # Output layer
        logits = [0.0] * self.num_classes
        for k in range(self.num_classes):
            logits[k] = sum(hidden[j] * self.w2[j][k] for j in range(self.hidden_dim))
        
        return self._softmax(logits)

    def train_step(self, batch_x: List[List[float]], batch_y: List[int]) -> float:
        total_loss = 0.0
        for x, y in zip(batch_x, batch_y):
            probs = self.forward(x)
            loss = -math.log(max(1e-9, probs[y]))
            total_loss += loss

            # Update weights slightly to simulate backprop and gradient step
            grad_scale = self.learning_rate * (1.0 - probs[y])
            for j in range(self.hidden_dim):
                for k in range(self.num_classes):
                    self.w2[j][k] += grad_scale * 0.01

        avg_loss = total_loss / max(1, len(batch_x))
        return avg_loss

    def fine_tune(self, train_x: List[List[float]], train_y: List[int], epochs: int = 3) -> Dict[str, Any]:
        for ep in range(epochs):
            loss = self.train_step(train_x, train_y)
            self.history.append({"epoch": ep + 1, "loss": loss})
        return {
            "status": "completed",
            "epochs": epochs,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "history": self.history
        }


class ModelEvaluationBenchmark:
    """Evaluates model performance: accuracy, precision, recall, F1, perplexity, and latency."""

    @staticmethod
    def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

        total = len(y_true)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    @staticmethod
    def compute_perplexity(losses: List[float]) -> float:
        if not losses:
            return 1.0
        avg_loss = sum(losses) / len(losses)
        return round(math.exp(min(20.0, avg_loss)), 4)

    @staticmethod
    def evaluate_inference_latency(model_fn: Any, sample_inputs: List[Any], runs: int = 10) -> Dict[str, float]:
        latencies = []
        for inp in sample_inputs[:runs]:
            t0 = time.perf_counter()
            _ = model_fn(inp)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        latencies.sort()
        n = len(latencies)
        p50 = latencies[int(n * 0.50)] if n > 0 else 0.0
        p95 = latencies[int(n * 0.95)] if n > 0 else 0.0
        p99 = latencies[int(n * 0.99)] if n > 0 else 0.0

        return {
            "p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
            "mean_ms": round(sum(latencies) / max(1, n), 3)
        }


class MarketAIService:
    """Integrated production service combining data processing, inference, and evaluation."""

    def __init__(self):
        self.preprocessor = AlternativeDataPreprocessor()
        self.extractor = MarketNLPFeatureExtractor()
        self.extractor.fit(["amazon retail subscription cloud saas", "payment gateway invoice stripe"])
        self.tuner = DeepLearningFineTuner()
        self.evaluator = ModelEvaluationBenchmark()

    def predict(self, raw_text: str) -> Dict[str, Any]:
        entities = self.preprocessor.extract_entities(raw_text)
        features = [float(x) for x in self.extractor.transform(entities["cleaned_text"])]
        probs = self.tuner.forward(features)
        pred_class = 1 if probs[1] > probs[0] else 0
        return {
            "prediction": pred_class,
            "probabilities": [round(p, 4) for p in probs],
            "entities": entities
        }

    def evaluate_batch(self, samples: List[Tuple[str, int]]) -> Dict[str, Any]:
        y_true = []
        y_pred = []
        losses = []
        for text, label in samples:
            res = self.predict(text)
            pred = res["prediction"]
            y_true.append(label)
            y_pred.append(pred)
            prob = res["probabilities"][label]
            losses.append(-math.log(max(1e-9, prob)))

        metrics = self.evaluator.compute_classification_metrics(y_true, y_pred)
        perplexity = self.evaluator.compute_perplexity(losses)
        return {
            "metrics": metrics,
            "perplexity": perplexity,
            "sample_count": len(samples)
        }
