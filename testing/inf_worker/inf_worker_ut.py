from . import worker_test_definition as iw
from datetime import datetime as dt
import numpy as np
import json
import csv
from pathlib import Path
import threading
import time

b_frame = np.zeros((640, 640, 3), dtype=np.uint8)
w_frame = np.full((640, 640, 3), 255, dtype=np.uint8)
frames = []
new_recs = []

with open("testing/inf_worker/inf_worker_test_cases.json", "r") as f:
    test_cases = json.load(f)

for tc in test_cases:
    results = tc.copy()

    iw.SKIP_FRAMES = tc["skip"]

    # (Matt): goal is to only read alternating frame colors once per skip
    for i in range(0, tc["num_imgs"], tc["skip"]):
        if len(frames) == 0 or np.array_equal(frames[-1][0, 0], [255, 255, 255]):
            frames += [b_frame] * tc["skip"]
        else:
            frames += [w_frame] * tc["skip"]

    frames = frames[:tc["num_imgs"]]

    iw.stop_event = threading.Event()
    iw.latest_frame = [None]
    iw.latest_result = [None]
    iw.lock = threading.Lock()
    model_results = []

    print(f"TC: {tc["name"]}")

    worker = threading.Thread(target=iw.inference_worker, daemon=True)
    worker.start()

    # (Matt): this block is to warm up the inference model since the first frame takes the longest to complete (likely due to initialization overhead)
    with iw.lock:
        iw.latest_frame[0] = np.zeros((640, 640, 3), dtype=np.uint8)
    while iw.latest_result[0] is None:
        time.sleep(0.001)
    with iw.lock:
        iw.latest_result[0] = None  # resets it for the test case
        time.sleep(0.5)
        iw.latest_result[0] = None

    for f in frames:
        with iw.lock:
            # For testing
            print("locking latest_frame in test runner")
            iw.latest_frame[0] = f

        time.sleep(1 / 30)

        with iw.lock:
            # For testing
            print("locking latest_result in test runner")

            if iw.latest_result[0] is None:
                model_results.append((None, None))
            else:
                model_results.append(iw.latest_result[0])

    iw.stop_event.set()

    worker.join()

    print()

    act_colors = []
    act_inf_results = False

    for mr in model_results:
        if mr[0] is not None:
            if np.array_equal(mr[0][0, 0], [0, 0, 0]):
                act_colors.append("black")
            else:
                act_colors.append("white")

        # if inference runs once, it is true
        if mr[1] is not None:
            act_inf_results = True

    results["act_inf_ran"] = act_inf_results
    results["act_colors"] = act_colors

    check_colors = tc["exp_frames"] == act_colors
    check_inf_ran = tc["exp_inference_ran"] == act_inf_results

    if check_colors and check_inf_ran:
        results["status"] = "Passed"
    else:
        results["status"] = "Failed"

    new_recs.append(results)
    frames = []

res_path = Path("testing/inf_worker/results_inf_worker.csv")
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
        "Input No. of Frames",
        "Input Skip Rate",
        "Expected Inference Ran",
        "Expected Frame Colors",
        "Actual Inference Ran",
        "Actual Frame Colors",
        "Status",
    ])
    w.writerows(rec.values() for rec in new_recs)

    if res_path.exists() and old_recs:
        w.writerow([])
        w.writerows(old_recs)