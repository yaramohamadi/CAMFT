"""Model builder for adversarial JiT-DMF MeFT (CA-iMF) post-training.

The adversarial flow wrapper (``SiTMeFTAdversarialFlow``) is architecture
agnostic: its forward-time helpers build ``x_t=(1-t)*noise+t*x`` and predict the
target velocity via the shared iMeanFlow machinery. JiT-DMF is transport space
(``target_velocity_map_mode='transport'``), which matches that convention
directly. So we reuse the wrapper and only swap in the JiT backbone + JiT
discriminator.
"""

import dataclasses

from imf import iMeanFlow
from models import jit as jit_models
from models.jit_meft_discriminator import create_jit_meft_discriminator
from train_sit_meft_common import SiTMeFTAdversarialFlow


def create_models(config, *, retention_weight=0.0):
    """Create the JiT-DMF MeFT generator wrapper and matching scalar D."""
    model_config = config.model.to_dict()
    valid_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    model_config = {
        key: value for key, value in model_config.items() if key in valid_keys
    }
    model = SiTMeFTAdversarialFlow(
        **model_config,
        compute_adversarial_retention=float(retention_weight) > 0.0,
    )
    model_str = str(config.model.model_str)
    if not model_str.startswith("imfJiT_DMF_"):
        raise ValueError(
            "Expected model.model_str=imfJiT_DMF_* from a JiT MeFT checkpoint; "
            f"got {model_str!r}."
        )
    net_fn = jit_models.JiT_models[model_str]
    generator_net = net_fn(
        name="net",
        num_classes=int(config.model.num_classes),
        use_null_class=bool(config.model.target_use_null_class),
        time_conditioning_mode=str(
            config.model.get("time_conditioning_mode", "split")
        ),
        eval=False,
    )
    return model, generator_net, create_jit_meft_discriminator(generator_net)
