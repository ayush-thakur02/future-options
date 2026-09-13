"""The trained ML forecaster.

Labelling, validation, models, calibration, inference, and training all live in
this one folder because they are one thing: a change to how labels are built is a
change to what the model learns, and keeping them apart invites the two to drift.
The seam that matters is the *pack boundary* — everything it publishes to the
rest of the platform is ``Predictor``.
"""

from .predictor import Predictor
from .trainer import HorizonResult, Trainer, load_artifact

__all__ = ["HorizonResult", "Predictor", "Trainer", "load_artifact"]
