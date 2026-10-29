"""Token pricing and cost attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .types import Usage


@dataclass(frozen=True)
class Price:
    input_per_1k_usd: float
    output_per_1k_usd: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.prompt_tokens / 1000.0 * self.input_per_1k_usd
            + usage.completion_tokens / 1000.0 * self.output_per_1k_usd
        )


class Pricebook:
    """Price lookup keyed by (provider, model), with a provider default."""

    def __init__(self, prices: Dict[Tuple[str, str], Price], defaults: Dict[str, Price]) -> None:
        self._prices = dict(prices)
        self._defaults = dict(defaults)

    def price_for(self, provider: str, model: str) -> Price:
        if (provider, model) in self._prices:
            return self._prices[(provider, model)]
        if provider in self._defaults:
            return self._defaults[provider]
        return Price(input_per_1k_usd=0.0, output_per_1k_usd=0.0)

    def cost(self, provider: str, model: str, usage: Usage) -> float:
        return self.price_for(provider, model).cost(usage)


def default_pricebook() -> Pricebook:
    prices = {
        ("gemini", "gemini-1.5-pro"): Price(0.00125, 0.005),
        ("gemini", "gemini-1.5-flash"): Price(0.000075, 0.0003),
        ("openai", "gpt-4o"): Price(0.0025, 0.01),
        ("openai", "gpt-4o-mini"): Price(0.00015, 0.0006),
        ("anthropic", "claude-3-5-sonnet"): Price(0.003, 0.015),
        ("fake-primary", "fake-large"): Price(0.0005, 0.0015),
        ("fake-fallback", "fake-small"): Price(0.0001, 0.0003),
    }
    defaults = {
        "gemini": Price(0.001, 0.004),
        "openai": Price(0.002, 0.008),
        "anthropic": Price(0.003, 0.015),
        "fake-primary": Price(0.0005, 0.0015),
        "fake-fallback": Price(0.0001, 0.0003),
    }
    return Pricebook(prices, defaults)
