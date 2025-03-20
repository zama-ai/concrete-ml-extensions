import pytest
import concrete_ml_extensions as fhext
import numpy as np
import time
import json

EXPECT_MSBS_CORRECT = 12
EXPECT_LST_MSB_CORRECT_FRACTION = 1 / 100  # 1 out of 100 incorrect


@pytest.fixture(scope="session")
def correctness_assumption():
    return (EXPECT_MSBS_CORRECT, EXPECT_LST_MSB_CORRECT_FRACTION)


class Timing:
    def __init__(self, message=""):
        self.message = message

    def __enter__(self):
        print(f"Starting {self.message}")
        self.start = time.time()

    def __exit__(self, *args, **kwargs):
        end = time.time()
        print(f"{self.message} done in {end - self.start} seconds")
