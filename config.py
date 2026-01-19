import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Updated to point to Datasets folder with hierarchical structure
DATA_ROOT = os.path.join(BASE_DIR, "Datasets")
PLANNING_ROOT = os.path.join(BASE_DIR, "storage", "plans")
