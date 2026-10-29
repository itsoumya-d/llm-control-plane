"""Versioned model aliases and deterministic failover order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .types import RouteTarget


@dataclass(frozen=True)
class Route:
    alias: str
    targets: List[RouteTarget]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(f"route {self.alias!r} needs at least one target")


class RoutingTable:
    def __init__(self, routes: Iterable[Route], default_alias: Optional[str] = None) -> None:
        self._routes: Dict[str, Route] = {route.alias: route for route in routes}
        if not self._routes:
            raise ValueError("routing table cannot be empty")
        self._default_alias = default_alias or next(iter(self._routes))
        if self._default_alias not in self._routes:
            raise ValueError(f"default alias {self._default_alias!r} is not configured")

    def resolve(self, alias: Optional[str]) -> List[RouteTarget]:
        key = alias or self._default_alias
        route = self._routes.get(key)
        if route is None:
            raise KeyError(f"unknown model alias {key!r}")
        return list(route.targets)

    def aliases(self) -> List[str]:
        return sorted(self._routes)

    @classmethod
    def from_dict(cls, data: Dict[str, List[Dict[str, str]]], default_alias: Optional[str] = None) -> "RoutingTable":
        routes = [
            Route(
                alias=alias,
                targets=[RouteTarget(provider=item["provider"], model=item["model"]) for item in targets],
            )
            for alias, targets in data.items()
        ]
        return cls(routes, default_alias=default_alias)
