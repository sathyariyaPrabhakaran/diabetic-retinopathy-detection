"""Compatibility shim for routers saved before the project was packaged as src."""
from src.router import LearnedRouter, routing_features

__all__ = ["LearnedRouter", "routing_features"]
