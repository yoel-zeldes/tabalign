import os
from joblib import Memory

OUTPUT_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
)
memory = Memory(OUTPUT_DIR, verbose=1)
