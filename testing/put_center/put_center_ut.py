import pothole_detection as pd
from datetime import datetime as dt
import numpy as np
import json
import csv
from pathlib import Path
import cv2

#same font used for function calls in source file
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 1

with open("testing/put_center/put_center_test_cases.json", "r") as f:
    test_cases = json.load(f)

new_recs = []

for tc in test_cases:
    results = tc.copy()

    img_w = tc["img_w"]
    img_h = tc["img_h"]
    text = tc["text"]
    cx = tc["cx"]
    y = tc["y"]     # baseline of text
    color = tc["color"][::-1]
    if tc["thickness"]:
        thickness = tc["thickness"]
    else:
        thickness = 1
        results["thickness"] = "Default"

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # get text area (width, height)
    tw, th = cv2.getTextSize(text, FONT, SCALE, thickness)[0]

    left = cx - (tw // 2)
    right = cx + (tw // 2)
    top = y - th

    # region inside boundary before change
    roi_before = img[top:y, left:right].copy()

    if tc["thickness"]:
        pd.put_center(img, text, cx, y, FONT, SCALE, color, thickness)
    else:
        pd.put_center(img, text, cx, y, FONT, SCALE, color)

    roi_changed = not np.array_equal(roi_before, img[top:y, left:right])
    results["act_change"] = roi_changed

    if tc["exp_change"] == roi_changed:
        results["passed"] = "Passed"
    else:
        results["passed"] = "Failed"

    new_recs.append(results)
    results = None

res_path = Path("testing/put_center/results_put_center.csv")
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
        "Text",
        "CX",
        "Y",
        "Color",
        "Thickness",
        "Expected Change",
        "Actual Change",
        "Status",
    ])
    w.writerows(rec.values() for rec in new_recs)

    if res_path.exists() and old_recs:
        w.writerow([])
        w.writerows(old_recs)
