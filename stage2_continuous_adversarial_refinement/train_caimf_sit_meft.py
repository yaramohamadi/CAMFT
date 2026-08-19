"""Finite-interval CA-iMF adapter for SiT-DMF MeFT checkpoints."""

import train_caimf as _base

from caimf_sit_meft import finite_fake_logit, finite_interval_logits
from train_sit_meft_common import create_models


def _create_models(config):
    model, _, discriminator = create_models(
        config, retention_weight=float(config.caimf.lambda_imf)
    )
    return model, discriminator


def train_and_evaluate(config, workdir):
    # Patch only this isolated entry point; existing main_caimf.py is unchanged.
    _base._create_models = _create_models
    _base.finite_interval_logits = finite_interval_logits
    _base.finite_fake_logit = finite_fake_logit
    return _base.train_and_evaluate(config, workdir)
