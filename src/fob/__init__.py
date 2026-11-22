"""Focus-on-Background (FoB) modules.

* :mod:`src.fob.bppc`     - Background Prompt Prototype Construction
* :mod:`src.fob.bcm`      - Background Context Modeling
* :mod:`src.fob.spr`      - Structure-guided Prompt Refinement
* :mod:`src.fob.adaptive` - Adaptive prompt configuration
* :mod:`src.fob.fob_pipeline` - End-to-end FoB / Enhanced FoB pipeline
"""

from src.fob.adaptive import get_adaptive_config
from src.fob.bcm import BackgroundContextModeling
from src.fob.bppc import BackgroundPromptPrototype
from src.fob.spr import StructureGuidedRefinement

__all__ = [
    "BackgroundContextModeling",
    "BackgroundPromptPrototype",
    "StructureGuidedRefinement",
    "get_adaptive_config",
]
