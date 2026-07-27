import pothole_detection as pd
import numpy as np
import json
import csv
from datetime import datetime as dt
from pathlib import Path

#(Matt): read test cases from JSON and convert to python format
with open("testing/init_ipm_test_cases.json", "r") as f:
    test_cases = json.load(f)

new_recs = []

for tc in test_cases:
    results = tc.copy()
    hg_check = False
    isp_check = False
    bb_check = False   # DELETE?

    pd.IPM_SRC_RATIO = tc["src_ratio"]
    expected_src_points = np.float32(tc["exp_src_pts"])

    pd.init_ipm(tc["width"], tc["height"])

    results["src_points"] = pd.ipm_src_pts
    if np.allclose(pd.ipm_src_pts, expected_src_points):
        isp_check = True

    # (Matt): checks if it is set, it is proper size, and if no invalid numbers
    if (pd.homography is not None 
    and pd.homography.shape == (3,3)
    and np.all(np.isfinite(pd.homography))):
        results["homography"] = "set"
        hg_check = True
    elif pd.homography is not None:
            results["homography"] = "Not Set"
    elif pd.homography.shape != (3,3):
        results["homography"] = "Invalid Shape"
    else:
        results["homography"] = "NaN/Inf Found"

    if pd._bev_buf is not None:
        results["bev_buf"] = "Set"
        bb_check = True
    else:
        results["bev_buf"] = "Not Set"

    if isp_check and hg_check and bb_check:
        results["status"] = "Passed"
    else:
        results["status"] = "Failed"
    
    new_recs.append(results)
    pd.homography = None
    pd._bev_buf = None
    pd.ipm_src_pts = None

res_path = Path("testing/results_init_ipm.csv")
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
        "Input Width",
        "Input Height",
        "Src Ratio",
        "Expected Src Pts",
        "Actual Src Pts",
        "Homography",
        "BEV Buffer",
        "Status",
    ])
    w.writerows(rec.values() for rec in new_recs)

    if res_path.exists() and old_recs:
        w.writerow([])
        w.writerows(old_recs)
