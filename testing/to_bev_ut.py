import pothole_detection as pd
from datetime import datetime as dt
import numpy as np
import json
import csv
from pathlib import Path

with open("testing/to_bev_test_cases.json", "r") as f:
    test_cases = json.load(f)

new_recs = []

for tc in test_cases:
    results = tc.copy()

    if tc['homography']:
        pd.homography = np.float32(tc['homography'])
    else:
        pd.homography = None

    if tc['exp_bev_pt'] is not None:
        expected_pt = tuple(tc['exp_bev_pt'])
    else:
        expected_pt = None

    actual_pt = pd.to_bev(tc['x'], tc['y'])

    results['actual_pts'] = actual_pt

    if actual_pt == expected_pt:
        results['status'] = "Passed"
    else:
        results['status'] = "Failed"

    new_recs.append(results)
    results = None

res_path = Path("testing/results_to_bev.csv")
old_recs = []

if res_path.exists():
    with open(res_path, "r") as f:
        r = csv.reader(f)
        old_recs = list(r)

with open(res_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"Ran: {dt.now()}"])
    w.writerow([
        "Name",
        "Input X",
        "Input Y",
        "Homography Matrix",
        "Expected BEV Pt",
        "Actaul BEV Pt",
        "Status",
    ])
    w.writerows(rec.values() for rec in new_recs)

    if res_path.exists() and old_recs:
        w.writerow([])
        w.writerows(old_recs)
