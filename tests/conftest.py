import pytest
import concrete_ml_extensions as fhext
import numpy as np
import time
import json

class Timing:
    def __init__(self, message=""):
        self.message = message

    def __enter__(self):
        print(f"Starting {self.message}")
        self.start = time.time()

    def __exit__(self, *args, **kwargs):
        end = time.time()
        print(f"{self.message} done in {end - self.start} seconds")