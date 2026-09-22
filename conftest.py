"""Put the project root on sys.path so tests can import the qgc package."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
