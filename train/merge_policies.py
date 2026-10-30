"""Compatibility entry point for the packaged merge experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'policy-manifolds'))
from kan_search.merge_experiment import main

if __name__ == '__main__': main()
