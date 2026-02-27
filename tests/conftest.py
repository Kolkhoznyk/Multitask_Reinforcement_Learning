"""
Pytest configuration: add the project root to sys.path so all sub-packages
(MT10_SAC, MT3_SAC, utils, …) are importable from every test file.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
