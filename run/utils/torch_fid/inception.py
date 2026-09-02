"""Torch InceptionV3 for FID/IS features. Port of utils/jax_fid/inception.py.

The Flax module in utils/jax_fid/inception.py is itself a port of pytorch-fid's
FID InceptionV3 (weights from the TF FID graph, hand-mapped into Flax by
utils/jax_fid/cvt.py). This module goes back to the source: it wraps
``pytorch_fid.inception.fid_inception_v3()`` (same weights, downloaded from the
pytorch-fid release on first use) and reproduces the exact semantics of the
Flax module as used by utils/fid_util.py:

- no input resize and no ``transform_input``: inputs are expected to be
  ``(B, 3, 299, 299)`` float in ``[-1, 1]``, i.e. the output of
  ``pt.utils.torch_fid.resize.forward`` (this corresponds to pytorch-fid's
  ``resize_input=False, normalize_input=False``);
- pooled features are the global average pool of Mixed_7c, shape ``(B, 2048)``;
- logits are computed from the pooled features with the ``fc`` kernel WITHOUT
  the bias: the Flax module calls its Dense head with ``unbiased=True``
  (utils/jax_fid/inception.py:157), so we use ``F.linear(pooled, fc.weight)``.

Semantic deltas vs the Flax module:
- returns ``(pooled_features, logits)``; the spatial (sFID) branch of the Flax
  module is not ported (utils/fid_util.py discards it).
- inference only: parameters are frozen and the module is in eval mode.
"""

import torch
import torch.nn.functional as F
from torch import nn

from pytorch_fid.inception import fid_inception_v3


class InceptionV3(nn.Module):
    """FID InceptionV3 returning (2048-d pooled features, unbiased 1008-way logits)."""

    def __init__(self):
        super().__init__()
        self.net = fid_inception_v3()
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        """Run the FID Inception trunk.

        Args:
            x: ``(B, 3, 299, 299)`` float tensor in ``[-1, 1]`` (BCHW), as
                produced by ``pt.utils.torch_fid.resize.forward``.

        Returns:
            Tuple ``(pooled_features, logits)`` with shapes ``(B, 2048)`` and
            ``(B, 1008)``.
        """
        net = self.net
        x = net.Conv2d_1a_3x3(x)
        x = net.Conv2d_2a_3x3(x)
        x = net.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = net.Conv2d_3b_1x1(x)
        x = net.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = net.Mixed_5b(x)
        x = net.Mixed_5c(x)
        x = net.Mixed_5d(x)
        x = net.Mixed_6a(x)
        x = net.Mixed_6b(x)
        x = net.Mixed_6c(x)
        x = net.Mixed_6d(x)
        x = net.Mixed_6e(x)
        x = net.Mixed_7a(x)
        x = net.Mixed_7b(x)
        x = net.Mixed_7c(x)

        # Global average pooling to 2048-d features.
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        pooled_features = torch.flatten(x, 1)

        # Classifier head applied UNBIASED, matching the Flax Dense(unbiased=True).
        logits = F.linear(pooled_features, net.fc.weight)

        return pooled_features, logits
