"""CLI wrapper for src/models/groupnorm_swap.py main()."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.groupnorm_swap import main

if __name__ == "__main__":
    main()