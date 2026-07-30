import pothole_detection as pd
from datetime import datetime as dt
import numpy as np
import json
import csv
from pathlib import Path

with open("testing/alpha_rect/alpha_rect_test_cases.json", "r") as f:
    test_cases = json.load(f)

new_recs = []

for tc in test_cases:
    results = tc.copy()

    img = np.zeros((tc["img_h"], tc["img_w"], 3), dtype=np.uint8)
    color_in = tuple(tc["color"])
    exp_color = np.array(tc["exp_color"][::-1], dtype=np.uint8)

    #(Matt): region inside boundary before change and outside
    roi_before = img[tc["y1"]:tc["y2"], tc["x1"]:tc["x2"]].copy()
    out_before = img[tc["y2"]:, :].copy()

    if tc["alpha"]:
        pd.alpha_rect(img, tc["x1"], tc["y1"], tc["x2"], tc["y2"], tc["color"], tc["alpha"])
    else:
        results["alpha"] = "Default"
        pd.alpha_rect(img, tc["x1"], tc["y1"], tc["x2"], tc["y2"], tc["color"])

    #(Matt): OpenCV array returns colors as BGR so reverse it to get RGB
    results["act_color"]= img[tc["y1"], tc["x1"]][::-1]

    color_check = np.all(img[tc["y1"]:tc["y2"], tc["x1"]:tc["x2"]] == exp_color)
    if color_check:
        results["region"] = "True"
    else:
        results["region"] = "False"

    inside_check = not np.array_equal(img[tc["y1"]:tc["y2"], tc["x1"]:tc["x2"]], roi_before)

    outside_check = np.array_equal(img[tc["y2"]:, :], out_before)

    if color_check and inside_check and outside_check:
        results["status"] = "Passed"
    else:
        results["status"] = "Failed"

    new_recs.append(results)
    results = None

res_path = Path("testing/alpha_rect/results_alpha_rect.csv")
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
        "Input Image Width",
        "Input Image Height",
        "X1",
        "Y1",
        "X2",
        "Y2",
        "Color",
        "Alpha",
        "Expected Color",
        "Actual Color",
        "Correct Region",
        "Status",
    ])
    w.writerows(rec.values() for rec in new_recs)

    if res_path.exists() and old_recs:
        w.writerow([])
        w.writerows(old_recs)
