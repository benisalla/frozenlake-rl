import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
