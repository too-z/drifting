"""Feature Extractor for the drift loss"""

from __future__ import annotations

def build_activation_function(**_ignored):
  """Returns (activation_fn, variables)."""
  variables = {}
  def activation_fn(params, x, **_kwargs):
    # x is [B, 1, D] for tabular; flatten to the single "global" view.
    return {"global": x.reshape(x.shape[0], 1, -1)}
  return activation_fn, variables
