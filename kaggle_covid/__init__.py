import os

__version__ = "0.1.0"
__docs__ = "Tooling for Kaggle COVID detection"
__author__ = "Jiri Borovec"
__author_email__ = "jirka@pytorchlightning.ai"
__homepage__ = "https://github.com/Borda/kaggle_COVID-detection"
__license__ = "MIT"

_PATH_PACKAGE = os.path.realpath(os.path.dirname(__file__))
_PATH_PROJECT = os.path.dirname(_PATH_PACKAGE)

LABELS = (
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
)
LABELS_SHORT = (
    "negative",
    "typical",
    "indeterminate",
    "atypical",
)
