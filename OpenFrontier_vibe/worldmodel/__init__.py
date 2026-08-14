"""
Frontier-conditioned counterfactual world modeling for OpenFrontier.

Implements the design in ``frontier_conditioned_world_model_design.md``:
each persistent frontier is treated as a counterfactual action; a
goal-agnostic world model predicts a distribution over what exploration
through that frontier would reveal, a goal evaluator matches predictions
against the language goal, and an uncertainty/cost-aware ranker replaces
the myopic OpenFrontier utility.

Entry point: :func:`worldmodel.pipeline.build_pipeline`.
"""

from worldmodel.records import FutureHypothesis, WMPrediction, FrontierWMRecord

__all__ = ["FutureHypothesis", "WMPrediction", "FrontierWMRecord"]
