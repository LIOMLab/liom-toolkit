from __future__ import annotations
from .register import *
from .templating import *

try:
    import ants
except ImportError:
    raise ImportError("Please install ANTsPy to use the registration module of the LIOM toolkit.")
