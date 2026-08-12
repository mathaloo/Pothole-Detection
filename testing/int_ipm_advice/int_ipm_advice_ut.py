from src import pothole_detection as pd
from datetime import datetime as dt
import numpy as np
import json
import csv
from pathlib import Path

with open("testing/int_ipm_advice/int_ipm_advice_test_cases.json", "r") as f:
    test_cases = json.load(f)

new_recs = []

pd.IPM_SRC_RATIO = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
]

for tc in test_cases:
    results = tc.copy()

    w = tc["width"]
    h = tc["height"]
    pd.BEV_W = tc["bev_w"]
    pd.BEV_H = tc["bev_h"]

    xyxy_arr = np.array(tc["xyxy_arr"], dtype=int)
    conf_arr = np.array(tc["conf_arr"], dtype=np.float32)
    centers = np.array(tc["centers"], dtype=np.float32)

    ipm = pd.IPMTransform(w, h)
    bev_pts = ipm.to_bev_batch(centers)

    check_bev = True
    for b in bev_pts:
        bx, by = b
        if bx < 0 or bx > pd.BEV_W:
            check_bev = False
        if by < 0 or by > pd.BEV_H:
                    check_bev = False
    if check_bev:
        results["act_bev"] = "In Bounds"
    else:
        results["act_bev"] = "Out of Bounds"

    act_advice, act_lateral, act_dist  = pd.get_advice_ipm(xyxy_arr, conf_arr, bev_pts, w, ipm)

    results["act_advice"] = act_advice
    check_advice = tc["exp_advice"] == act_advice

    check_lat = False
    if act_lateral is not None:
        results["act_lateral"] = "Valid"
        check_lat = True
    else:
        results["act_lateral"] = "Invalid"

    check_dist = False
    if act_dist is not None:
        results["act_dist"] = "Valid"
        check_dist = True
    else:
        results["act_dist"] = "Invalid"

    if check_bev and check_advice and check_lat and check_dist:
        results["status"] = "Passed"
    else:
        results["status"] = "Failed"

    new_recs.append(results)

rec_path = Path("testing/int_ipm_advice/results_int_ipm_advice.csv")
old_recs = []

if rec_path.exists():
    with open(rec_path, "r") as f:
        r = csv.reader(f)
        old_recs = list(r)

with open(rec_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"Ran: {dt.now()}"])
    w.writerow([
        "Name",
        "Input Width",
        "Input Height",
        "Input BEV Width",
        "Input BEV Height",
        "XYXY Array",
        "Center Points",
        "Confidence Array",
        "Expected Advice",
        "Actual BEV Coords",
        "Actual Advice",
        "Actual Lateral",
        "Actual Dist",
        "Status"
    ])
    w.writerows(r.values() for r in new_recs)

    if len(old_recs) > 0:
        w.writerow()
        w.writerows(old_recs)