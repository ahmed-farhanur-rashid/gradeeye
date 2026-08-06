"""src.segmentation: lesion/structure segmentation for the 4-channel grader."""
from src.segmentation.ablation_3arm import (
    ARMS,
    collect_idrid_samples,
    collect_drive_samples,
)

__all__ = ["ARMS", "collect_idrid_samples", "collect_drive_samples"]
