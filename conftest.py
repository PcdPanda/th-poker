from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("./src/python"))
import pytest


@pytest.fixture
def int_fixture():
    return 3
