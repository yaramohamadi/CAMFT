"""AFM post-training adapter for target-domain SiT-DMF MeFT checkpoints."""

import train_afm as _base

from train_sit_meft_common import (
    create_models,
    make_afm_endpoint_terms,
    mask_sit_discriminator_grads,
)


def _create_models(config):
    return create_models(config, retention_weight=float(config.afm.lambda_imf))


def train_and_evaluate(config, workdir):
    # Patch only this isolated entry point; existing main_afm.py is unchanged.
    _base._create_models = _create_models
    _base._make_endpoint_terms = make_afm_endpoint_terms
    _base._mask_discriminator_grads = mask_sit_discriminator_grads
    return _base.train_and_evaluate(config, workdir)
