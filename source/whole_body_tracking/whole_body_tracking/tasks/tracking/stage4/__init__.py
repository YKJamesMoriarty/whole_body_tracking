"""Stage 4: Frozen multi-skill experts + trainable router (MoE)."""

from whole_body_tracking.tasks.tracking.stage4.skill_registry import (
    STAGE4_SKILL_MODEL_FILENAMES,
    resolve_stage4_model_paths,
)
from whole_body_tracking.tasks.tracking.stage4.amp_discriminator import AmpDiscriminator
