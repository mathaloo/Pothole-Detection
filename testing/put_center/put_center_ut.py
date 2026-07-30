import pothole_detection as pd
from datetime import datetime as dt
import numpy as np
import json
import csv
from pathlib import Path

with open("testing/put_center/put_center_test_cases.json", "r") as f:
    test_cases = json.load(f)

new_recs = []

for tc in test_cases:
    results = tc.copy()
    