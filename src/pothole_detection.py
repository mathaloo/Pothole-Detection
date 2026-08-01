import cv2
import time
import os
import threading
import numpy as np
import torch
from ultralytics import YOLO

torch.backends.cudnn.benchmark = True

MODEL_PATH = "../models/best.pt"
ENGINE_PATH = "../models/best.engine"
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
DEVICE = "cuda"
CAMERA_INDEX = 0
FRAME_W = 1280
FRAME_H = 720
BEV_W = 320
BEV_H = 400
MAX_FPS = 1000.0

IPM_SRC_RATIO = [
    [0.38, 0.58],
    [0.62, 0.58],
    [0.92, 0.95],
    [0.08, 0.95],
]

RED = (255, 60, 50)
GREEN = (80, 210, 100)
AMBER = (255, 185, 30)
WHITE = (245, 240, 240)
MUTED = (130, 110, 110)
PANEL = (24, 18, 18)
CYAN = (60, 210, 210)
DIM_RED = tuple(int(c * 0.35) for c in RED)
CYAN_DIM = tuple(int(c * 0.45) for c in CYAN)

FONT = cv2.FONT_HERSHEY_DUPLEX

BEV_SCALE = 0.28
BEV_SCALED_W = int(BEV_W * BEV_SCALE)
BEV_SCALED_H = int(BEV_H * BEV_SCALE)
BEV_CX = BEV_SCALED_W // 2

class IPMTransform:
    def __init__(self, fw, fh):
        self.bev_dst_pts = np.float32([
            [0, 0],
            [BEV_W, 0],
            [BEV_W, BEV_H],
            [0, BEV_H],
        ])
        self.ipm_src_pts = np.float32([[r[0] * fw, r[1] * fh] for r in IPM_SRC_RATIO])
        self.homography = cv2.getPerspectiveTransform(self.ipm_src_pts, self.bev_dst_pts)

    def to_bev_batch(self, pts):
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts_reshaped, self.homography)
        return out.reshape(-1, 2)

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.60):
    roi = img[y1:y2, x1:x2]
    if roi.size > 0:
        overlay = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0, dst=roi)

def put_center(img, text, cx, y, font, scale, color, thick=1):
    w = cv2.getTextSize(text, font, scale, thick)[0][0]
    cv2.putText(img, text, (cx - w // 2, y), font, scale, color, thick, cv2.LINE_AA)

def get_advice_ipm(xyxy_arr, conf_arr, bev_pts, frame_w, ipm):
    if len(xyxy_arr) > 0:
        best_idx = np.argmax(conf_arr)
        
        if ipm is not None and len(bev_pts) > best_idx:
            bx, by = bev_pts[best_idx]
            lateral = (bx / BEV_W) * 2.0 - 1.0
            dist = 1.0 - (by / BEV_H)
        else:
            x1, _, x2, _ = xyxy_arr[best_idx]
            cam_cx = (x1 + x2) / 2
            lateral = (cam_cx / frame_w) * 2.0 - 1.0
            dist = 0.5

        if lateral < -0.25:
            advice = "SHIFT RIGHT"
        elif lateral > 0.25:
            advice = "SHIFT LEFT"
        else:
            advice = "SLOW DOWN"

        return advice, lateral, dist
    return None, 0.0, 0.0

def draw_hud(frame, results, conf, fps, now, ipm):
    fh, fw = frame.shape[:2]
    out = cv2.convertScaleAbs(frame, alpha=0.86, beta=0)

    if ipm is not None:
        pts = ipm.ipm_src_pts.astype(np.int32)
        cv2.polylines(out, [pts], True, CYAN_DIM, 1)

    boxes = results.boxes
    n = len(boxes)
    detected = n > 0

    xyxy_arr = boxes.xyxy.cpu().numpy().astype(int) if n > 0 else np.empty((0, 4))
    conf_arr = boxes.conf.cpu().numpy() if n > 0 else np.empty((0,))

    bev_pts = np.empty((0, 2), dtype=np.float32)
    if n > 0 and ipm is not None:
        centers = np.column_stack(((xyxy_arr[:, 0] + xyxy_arr[:, 2]) / 2.0, xyxy_arr[:, 3]))
        bev_pts = ipm.to_bev_batch(centers)

    advice, lateral, dist = get_advice_ipm(xyxy_arr, conf_arr, bev_pts, fw, ipm)

    if n > 0:
        for (x1, y1, x2, y2), cf in zip(xyxy_arr, conf_arr):
            cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), DIM_RED, 1)
            cv2.rectangle(out, (x1, y1), (x2, y2), RED, 2)
            badge = f"{cf:.0%}"
            bw = cv2.getTextSize(badge, FONT, 0.52, 1)[0][0]
            alpha_rect(out, x1, y1 - 22, x1 + bw + 10, y1, PANEL)
            cv2.putText(out, badge, (x1 + 5, y1 - 6), FONT, 0.52, RED, 1, cv2.LINE_AA)

    bar_h = 62
    alpha_rect(out, 0, 0, fw, bar_h, PANEL, 0.82)
    cv2.line(out, (0, bar_h), (fw, bar_h), (52, 40, 40), 1)

    if detected:
        pulse = 0.55 + 0.45 * abs(np.sin(now * 4.5))
        stripe = (int(RED[0] * pulse), int(RED[1] * pulse), int(RED[2] * pulse))
        cv2.rectangle(out, (0, 0), (5, bar_h), stripe, -1)
        cv2.circle(out, (26, bar_h // 2), 7, RED, -1)
        cv2.putText(out, "POTHOLE DETECTED", (44, 24), FONT, 0.72, WHITE, 1, cv2.LINE_AA)
        if advice:
            pw = cv2.getTextSize(advice, FONT, 0.60, 1)[0][0]
            px, py = 44, 32
            alpha_rect(out, px - 2, py - 2, px + pw + 14, py + 22, RED, 0.22)
            cv2.rectangle(out, (px - 2, py - 2), (px + pw + 14, py + 22), RED, 1)
            cv2.putText(out, advice, (px + 5, py + 16), FONT, 0.60, RED, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(out, (0, 0), (5, bar_h), GREEN, -1)
        cv2.circle(out, (26, bar_h // 2), 7, GREEN, -1)
        cv2.putText(out, "Road Clear", (44, 40), FONT, 0.78, GREEN, 1, cv2.LINE_AA)

    if detected:
        gw, gh = 260, 80
        gx, gy = fw // 2 - gw // 2, bar_h + 8

        alpha_rect(out, gx, gy, gx + gw, gy + gh, PANEL, 0.80)
        cv2.rectangle(out, (gx, gy), (gx + gw, gy + gh), (56, 42, 42), 1)

        tx1, tx2 = gx + 18, gx + gw - 18
        tlen = tx2 - tx1
        ty = gy + 36
        cv2.line(out, (tx1, ty), (tx2, ty), (68, 50, 50), 3)

        mid = tx1 + tlen // 2
        sz = tlen // 6
        alpha_rect(out, mid - sz, ty - 6, mid + sz, ty + 6, GREEN, 0.18)
        cv2.line(out, (mid, ty - 12), (mid, ty + 12), (78, 60, 60), 2)

        clamp = max(tx1 + 10, min(tx2 - 10, mid + int(lateral * tlen // 2)))
        dot_c = RED if abs(lateral) > 0.25 else GREEN
        cv2.circle(out, (clamp, ty), 10, dot_c, -1)
        cv2.circle(out, (clamp, ty), 10, WHITE, 1)

        cv2.putText(out, "L", (tx1 - 16, ty + 5), FONT, 0.52, MUTED, 1, cv2.LINE_AA)
        cv2.putText(out, "R", (tx2 + 5, ty + 5), FONT, 0.52, MUTED, 1, cv2.LINE_AA)

        dist_label = "CLOSE" if dist < 0.33 else "MID" if dist < 0.66 else "FAR"
        pip_map = {"CLOSE": RED, "MID": AMBER, "FAR": GREEN}
        for i, (lbl, col) in enumerate(pip_map.items()):
            active = lbl == dist_label
            px_ = gx + 36 + i * 76
            py_ = gy + 60
            c = col if active else MUTED
            cv2.circle(out, (px_, py_), 5 if active else 3, c, -1)
            cv2.putText(out, lbl, (px_ + 9, py_ + 5), FONT, 0.40, c, 1, cv2.LINE_AA)

    if ipm is not None:
        bev_full = cv2.warpPerspective(frame, ipm.homography, (BEV_W, BEV_H), flags=cv2.INTER_NEAREST)
        bev_s = cv2.resize(bev_full, (BEV_SCALED_W, BEV_SCALED_H), interpolation=cv2.INTER_NEAREST)
        bev_s = cv2.convertScaleAbs(bev_s, alpha=0.72, beta=0)
        cv2.line(bev_s, (BEV_CX, 0), (BEV_CX, BEV_SCALED_H), (80, 60, 60), 1)

        if n > 0 and len(bev_pts) > 0:
            for bx, by in bev_pts:
                bx_ = int(bx * BEV_SCALE)
                by_ = int(by * BEV_SCALE)
                if 4 <= bx_ < BEV_SCALED_W - 4 and 4 <= by_ < BEV_SCALED_H - 4:
                    cv2.circle(bev_s, (bx_, by_), 7, RED, -1)
                    cv2.circle(bev_s, (bx_, by_), 7, WHITE, 1)

        ox = fw - BEV_SCALED_W - 12
        oy = bar_h + 10
        alpha_rect(out, ox - 6, oy - 6, ox + BEV_SCALED_W + 6, oy + BEV_SCALED_H + 20, PANEL, 0.82)
        cv2.rectangle(out, (ox - 6, oy - 6), (ox + BEV_SCALED_W + 6, oy + BEV_SCALED_H + 20), (56, 42, 42), 1)
        out[oy:oy + BEV_SCALED_H, ox:ox + BEV_SCALED_W] = bev_s
        put_center(out, "BIRD'S EYE", ox + BEV_SCALED_W // 2, oy + BEV_SCALED_H + 14, FONT, 0.38, CYAN)

    strip_h = 34
    sy = fh - strip_h
    alpha_rect(out, 0, sy, fw, fh, PANEL, 0.82)
    cv2.line(out, (0, sy), (fw, sy), (56, 42, 42), 1)

    items = [
        (f"FPS  {fps:.0f}", WHITE),
        (f"CONF  {conf:.0%}", WHITE),
        (f"DETECTIONS  {n}", RED if n > 0 else MUTED),
        ("+/-  conf", MUTED),
        ("R  reset IPM", MUTED),
        ("Q  quit", MUTED),
    ]
    slot = fw // len(items)
    for i, (label, color) in enumerate(items):
        put_center(out, label, slot * i + slot // 2, fh - 10, FONT, 0.44, color)
        if i:
            cv2.line(out, (slot * i, sy + 5), (slot * i, fh - 5), (56, 42, 42), 1)
    return out

def main():
    cv2.setNumThreads(3)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, MAX_FPS)

        WIN = "Pothole Detection"
        cv2.namedWindow(WIN, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WIN, FRAME_W, FRAME_H)

        conf = CONF_THRESHOLD
        fc = 0
        fps = 0.0
        fps_t = time.time()
        ipm = None

        ret, first_frame = cap.read()
        if ret:
            fh_, fw_ = first_frame.shape[:2]
            ipm = IPMTransform(fw_, fh_)
            print(f"IPM ready — {fw_}x{fh_} | R resets | Q quits\n")

        stop_event = threading.Event()
        frame_condition = threading.Condition()
        latest_frame = None
        latest_result = None

        def inference_worker():
            nonlocal latest_result, latest_frame
            model = None
            if os.path.exists(ENGINE_PATH):
                model = YOLO(ENGINE_PATH, task="detect")
                print(f"Loaded TensorRT engine: {ENGINE_PATH}")
            elif os.path.exists(MODEL_PATH):
                print(f"No {ENGINE_PATH} found, exporting now...")
                try:
                    export_model = YOLO(MODEL_PATH, task="detect")
                    export_model.export(format="engine", imgsz=IMAGE_SIZE, half=True, device=DEVICE)
                    model = YOLO(ENGINE_PATH, task="detect")
                    print(f"Exported and loaded {ENGINE_PATH}")
                except Exception as e:
                    print(f"TensorRT export failed ({e}), falling back to {MODEL_PATH}")
                    model = YOLO(MODEL_PATH, task="detect")
            else:
                print(f"Model not found: {MODEL_PATH}")
                stop_event.set()

            if model is not None:
                warmup_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                model.predict(source=warmup_frame, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)

                while not stop_event.is_set():
                    with frame_condition:
                        while latest_frame is None and not stop_event.is_set():
                            frame_condition.wait()
                        if stop_event.is_set():
                            break
                        frame = latest_frame
                        c = conf

                    result = model.predict(
                        source=frame,
                        conf=c,
                        iou=IOU_THRESHOLD,
                        imgsz=IMAGE_SIZE,
                        device=DEVICE,
                        verbose=False,
                    )[0]

                    with frame_condition:
                        latest_result = result

        worker = threading.Thread(target=inference_worker, daemon=True)
        worker.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with frame_condition:
                latest_frame = frame
                results = latest_result
                frame_condition.notify()

            fc += 1
            if fc % 10 == 0:
                fps = 10 / (time.time() - fps_t)
                fps_t = time.time()

            now = time.time()

            if results is not None:
                out = draw_hud(frame, results, conf, fps, now, ipm)
            else:
                out = frame

            cv2.imshow(WIN, out)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif fc % 10 == 0 and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif k == ord('r'):
                fw_curr = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh_curr = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ipm = IPMTransform(fw_curr, fh_curr)
                print("IPM reset")
            elif k in (ord('+'), ord('=')):
                conf = min(0.95, conf + 0.05)
            elif k == ord('-'):
                conf = max(0.05, conf - 0.05)

        stop_event.set()
        with frame_condition:
            frame_condition.notify_all()
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Camera {CAMERA_INDEX} unavailable")

if __name__ == "__main__":
    main()
