import pothole_detection as pd
from datetime import datetime as dt
import numpy as np
import json
import csv
from pathlib import Path

#(Matt): used to simulate yolo prediction results (boxes)
class DummyBox:
    def __init__(self, coords, conf):
        self.xyxy = np.array([coords], dtype=float)
        self.conf = conf

#(Matt): used as stub to replace to_bev calls in get_advice_ipm()
def fake_to_bev(x, y):
   #(Matt): for testing when bpt not set by to_bev()
   if x > 1000 or y > 1000:
       return None

   return x, y

pd.to_bev = fake_to_bev 

with open("testing/get_advice_test_cases.json", "r") as f:
    test_cases = json.load(f)

new_recs = []

for tc in test_cases:
    adv_check = False
    lat_check = False
    dist_check = False

    results = tc.copy()

    boxes = []
    for b in tc["boxes"]:
        boxes.append(DummyBox(b["coords"], b["conf"]))
    
    pd.homography = tc["homography"]
    pd.BEV_W = tc["bevw"]
    pd.BEV_H = tc["bevh"]

    act_advice, act_lateral, act_dist  = pd.get_advice_ipm(boxes, tc["width"])

    if tc["homography"] is None:
        results["homography"] = "None"

    results["act_advice"] = act_advice
    if tc["exp_advice"] == act_advice:
        adv_check = True

    results["act_lateral"] = act_lateral
    if tc["exp_lateral"] == act_lateral:
        lat_check = True

    results["act_dist"] = act_dist
    if tc["exp_dist"] == act_dist:
        dist_check = True

    if adv_check and lat_check and dist_check:
        results["status"] = "Passed"
    else:
        results["status"] = "Failed"

    new_recs.append(results)
    results = None

res_path = Path("testing/results_get_advice.csv")
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
        "Input Bounding Boxes",
        "Input BEV Width",
        "Input BEV Height",
        "Homography",
        "Expected Advice",
        "Expected Lateral",
        "Expected Dist",
        "Actual Advice",
        "Actual Lateral",
        "Actual Dist",
        "Status"
    ])
    w.writerows(rec.values() for rec in new_recs)

    if res_path.exists():
        w.writerow([])
        w.writerows(old_recs)
    