# NOTE(griffin): pip install ultralytics opencv-python numpy

import cv2
import time
import os
import threading
import numpy as np
from ultralytics import YOLO

MODEL_PATH     = "best.pt"
CONF_THRESHOLD = 0.45
IOU_THRESHOLD  = 0.45
IMAGE_SIZE     = 640   
DEVICE         = 0 
CAMERA_INDEX   = 0
ALERT_COOLDOWN = 2.0
FRAME_W        = 1280
FRAME_H        = 720
SKIP_FRAMES    = 1        

BEV_W = 320
BEV_H = 400

# NOTE(griffin): Kinda a hack, but just made the IPM ROI static. Can adjust if need.
IPM_SRC_RATIO = [
    [0.38, 0.58],
    [0.62, 0.58],
    [0.92, 0.95],
    [0.08, 0.95],
]

RED   = (50,  60,  255)
GREEN = (100, 210, 80)
AMBER = (30,  185, 255)
WHITE = (240, 240, 245)
MUTED = (110, 110, 130)
PANEL = (18,  18,  24)
CYAN  = (210, 210, 60)
FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTS = cv2.FONT_HERSHEY_SIMPLEX

bev_dst_pts = np.float32([
    [0,     0],
    [BEV_W, 0],
    [BEV_W, BEV_H],
    [0,     BEV_H],
])

homography  = None
ipm_src_pts = None

_bev_scale  = 0.28
_bev_w      = int(BEV_W * _bev_scale)
_bev_h      = int(BEV_H * _bev_scale)
_bev_cx     = _bev_w // 2
_bev_buf    = None  


def init_ipm(fw, fh):
    global homography, ipm_src_pts, _bev_buf
    ipm_src_pts = np.float32([
        [r[0] * fw, r[1] * fh] for r in IPM_SRC_RATIO
    ])
    homography = cv2.getPerspectiveTransform(ipm_src_pts, bev_dst_pts)
    _bev_buf   = np.zeros((_bev_h, _bev_w, 3), dtype=np.uint8)


def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.60):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    cv2.addWeighted(
        np.full_like(roi, color[::-1]),
        alpha, roi, 1 - alpha, 0, roi
    )
    img[y1:y2, x1:x2] = roi


def put_center(img, text, cx, y, font, scale, color, thick=1):
    w = cv2.getTextSize(text, font, scale, thick)[0][0]
    cv2.putText(img, text, (cx - w // 2, y), font, scale, color, thick, cv2.LINE_AA)


def put_right(img, text, rx, y, font, scale, color, thick=1):
    w = cv2.getTextSize(text, font, scale, thick)[0][0]
    cv2.putText(img, text, (rx - w, y), font, scale, color, thick, cv2.LINE_AA)


def to_bev(x, y):
    if homography is None:
        return None
    pt  = np.float32([[[x, y]]])
    out = cv2.perspectiveTransform(pt, homography)
    return int(out[0][0][0]), int(out[0][0][1])


def get_advice_ipm(boxes, frame_w):
    if not len(boxes):
        return None, 0.0, 0.0

    best = max(boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = best.xyxy[0].tolist()
    cam_cx = (x1 + x2) / 2
    cam_by = y2

    if homography is not None:
        bpt = to_bev(cam_cx, cam_by)
        if bpt:
            bx, by   = bpt
            lateral  = (bx / BEV_W) * 2.0 - 1.0
            dist     = 1.0 - (by / BEV_H)
        else:
            lateral  = (cam_cx / frame_w) * 2.0 - 1.0
            dist     = 0.5
    else:
        lateral = (cam_cx / frame_w) * 2.0 - 1.0
        dist    = 0.5

    if lateral < -0.25:
        advice = "SHIFT RIGHT"
    elif lateral > 0.25:
        advice = "SHIFT LEFT"
    else:
        advice = "SLOW DOWN"

    return advice, lateral, dist


def draw_hud(frame, results, conf, fps, now):
    fh, fw = frame.shape[:2]

    out = cv2.addWeighted(frame, 0.86, np.zeros_like(frame), 0.14, 0)

    if ipm_src_pts is not None:
        pts = ipm_src_pts.astype(np.int32)
        cv2.polylines(out, [pts], True,
                      tuple(int(c * 0.45) for c in CYAN[::-1]), 1)

    boxes    = results.boxes
    n        = len(boxes)
    detected = n > 0
    advice, lateral, dist = get_advice_ipm(boxes, fw)

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cf    = float(box.conf)
        dim   = tuple(int(c * 0.35) for c in RED[::-1])
        cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), dim, 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), RED[::-1], 2)
        badge = f"{cf:.0%}"
        bw    = cv2.getTextSize(badge, FONTS, 0.52, 1)[0][0]
        alpha_rect(out, x1, y1 - 22, x1 + bw + 10, y1, PANEL)
        cv2.putText(out, badge, (x1 + 5, y1 - 6),
                    FONTS, 0.52, RED[::-1], 1, cv2.LINE_AA)

    bar_h = 62
    alpha_rect(out, 0, 0, fw, bar_h, PANEL, 0.82)
    cv2.line(out, (0, bar_h), (fw, bar_h), (40, 40, 52), 1)

    if detected:
        pulse  = 0.55 + 0.45 * abs(np.sin(now * 4.5))
        stripe = tuple(int(c * pulse) for c in RED[::-1])
        cv2.rectangle(out, (0, 0), (5, bar_h), stripe, -1)
        cv2.circle(out, (26, bar_h // 2), 7, RED[::-1], -1)
        cv2.putText(out, "POTHOLE DETECTED", (44, 24),
                    FONT, 0.72, WHITE[::-1], 1, cv2.LINE_AA)
        if advice:
            pw = cv2.getTextSize(advice, FONTS, 0.60, 1)[0][0]
            px, py = 44, 32
            alpha_rect(out, px - 2, py - 2, px + pw + 14, py + 22, RED, 0.22)
            cv2.rectangle(out, (px - 2, py - 2),
                          (px + pw + 14, py + 22), RED[::-1], 1)
            cv2.putText(out, advice, (px + 5, py + 16),
                        FONTS, 0.60, RED[::-1], 1, cv2.LINE_AA)
    else:
        cv2.rectangle(out, (0, 0), (5, bar_h), GREEN[::-1], -1)
        cv2.circle(out, (26, bar_h // 2), 7, GREEN[::-1], -1)
        cv2.putText(out, "Road Clear", (44, 40),
                    FONT, 0.78, GREEN[::-1], 1, cv2.LINE_AA)

    if detected:
        gw = 260
        gh = 80
        gx = fw // 2 - gw // 2
        gy = bar_h + 8

        alpha_rect(out, gx, gy, gx + gw, gy + gh, PANEL, 0.80)
        cv2.rectangle(out, (gx, gy), (gx + gw, gy + gh), (42, 42, 56), 1)

        tx1, tx2 = gx + 18, gx + gw - 18
        tlen = tx2 - tx1
        ty   = gy + 36
        cv2.line(out, (tx1, ty), (tx2, ty), (50, 50, 68), 3)

        mid = tx1 + tlen // 2
        sz  = tlen // 6
        alpha_rect(out, mid - sz, ty - 6, mid + sz, ty + 6, GREEN, 0.18)
        cv2.line(out, (mid, ty - 12), (mid, ty + 12), (60, 60, 78), 2)

        clamp = max(tx1 + 10, min(tx2 - 10, mid + int(lateral * tlen // 2)))
        dot_c = RED[::-1] if abs(lateral) > 0.25 else GREEN[::-1]
        cv2.circle(out, (clamp, ty), 10, dot_c, -1)
        cv2.circle(out, (clamp, ty), 10, WHITE[::-1], 1)

        cv2.putText(out, "L", (tx1 - 16, ty + 5),
                    FONTS, 0.52, MUTED[::-1], 1, cv2.LINE_AA)
        cv2.putText(out, "R", (tx2 + 5, ty + 5),
                    FONTS, 0.52, MUTED[::-1], 1, cv2.LINE_AA)

        dist_label = "CLOSE" if dist < 0.33 else "MID" if dist < 0.66 else "FAR"
        pip_map    = {"CLOSE": RED, "MID": AMBER, "FAR": GREEN}
        for i, (lbl, col) in enumerate(pip_map.items()):
            active = lbl == dist_label
            px_    = gx + 36 + i * 76
            py_    = gy + 60
            c      = col[::-1] if active else MUTED[::-1]
            cv2.circle(out, (px_, py_), 5 if active else 3, c, -1)
            cv2.putText(out, lbl, (px_ + 9, py_ + 5),
                        FONTS, 0.40, c, 1, cv2.LINE_AA)

    if homography is not None and _bev_buf is not None:
        bev_full = cv2.warpPerspective(frame, homography, (BEV_W, BEV_H),
                                       flags=cv2.INTER_NEAREST)
        bev_s    = cv2.resize(bev_full, (_bev_w, _bev_h),
                              interpolation=cv2.INTER_NEAREST)
        cv2.addWeighted(bev_s, 0.72, np.zeros_like(bev_s), 0.28, 0, bev_s)
        cv2.line(bev_s, (_bev_cx, 0), (_bev_cx, _bev_h), (60, 60, 80), 1)

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bpt = to_bev((x1 + x2) / 2, y2)
            if bpt:
                bx_ = int(bpt[0] * _bev_scale)
                by_ = int(bpt[1] * _bev_scale)
                if 4 <= bx_ < _bev_w - 4 and 4 <= by_ < _bev_h - 4:
                    cv2.circle(bev_s, (bx_, by_), 7, RED[::-1], -1)
                    cv2.circle(bev_s, (bx_, by_), 7, WHITE[::-1], 1)

        ox = fw - _bev_w - 12
        oy = bar_h + 10
        alpha_rect(out, ox - 6, oy - 6, ox + _bev_w + 6, oy + _bev_h + 20,
                   PANEL, 0.82)
        cv2.rectangle(out, (ox - 6, oy - 6),
                      (ox + _bev_w + 6, oy + _bev_h + 20), (42, 42, 56), 1)
        out[oy:oy + _bev_h, ox:ox + _bev_w] = bev_s
        put_center(out, "BIRD'S EYE", ox + _bev_w // 2, oy + _bev_h + 14,
                   FONTS, 0.38, CYAN[::-1])

    strip_h = 34
    sy      = fh - strip_h
    alpha_rect(out, 0, sy, fw, fh, PANEL, 0.82)
    cv2.line(out, (0, sy), (fw, sy), (42, 42, 56), 1)

    items = [
        (f"FPS  {fps:.0f}",       WHITE),
        (f"CONF  {conf:.0%}",     WHITE),
        (f"DETECTIONS  {n}",      RED if n > 0 else MUTED),
        ("+/-  conf",             MUTED),
        ("R  reset IPM",          MUTED),
        ("Q  quit",               MUTED),
    ]
    slot = fw // len(items)
    for i, (label, color) in enumerate(items):
        put_center(out, label, slot * i + slot // 2, fh - 10,
                   FONTS, 0.44, color[::-1])
        if i:
            cv2.line(out, (slot * i, sy + 5), (slot * i, fh - 5),
                     (42, 42, 56), 1)

    return out


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Camera {CAMERA_INDEX} unavailable")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    WIN = "Pothole Detection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, FRAME_W, FRAME_H)

    conf       = CONF_THRESHOLD
    last_chime = 0.0
    fc         = 0
    fps        = 0.0
    fps_t      = time.time()

    ret, first_frame = cap.read()
    if ret:
        fh_, fw_ = first_frame.shape[:2]
        init_ipm(fw_, fh_)
        print(f"IPM ready — {fw_}x{fh_} | R resets | Q quits\n")

    stop_event    = threading.Event()
    latest_frame  = [None]
    latest_result = [None]
    lock          = threading.Lock()

    def inference_worker():
        local_fc = 0
        while not stop_event.is_set():
            with lock:
                frame = latest_frame[0]
            if frame is None:
                time.sleep(0.001)
                continue
            local_fc += 1
            if local_fc % max(1, SKIP_FRAMES) != 0:
                time.sleep(0.001)
                continue
            result = model.predict(
                source=frame,
                conf=conf,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
                half=True,
            )[0]
            with lock:
                latest_result[0] = (frame, result)

    # NOTE(griffin): Had to add worker thread bc inference was so slow it blocked operations.
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            latest_frame[0] = frame
            payload         = latest_result[0]

        fc += 1
        if fc % 10 == 0:
            fps   = 10 / (time.time() - fps_t)
            fps_t = time.time()

        now = time.time()

        if payload is not None:
            render_frame, results = payload
            out = draw_hud(render_frame, results, conf, fps, now)
            if len(results.boxes) and (now - last_chime) > ALERT_COOLDOWN:
                os.system("afplay /System/Library/Sounds/Ping.aiff &")
                last_chime = now
        else:
            out = frame.copy()

        cv2.imshow(WIN, out)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif k == ord('r'):
            init_ipm(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print("IPM reset")
        elif k in (ord('+'), ord('=')):
            conf = min(0.95, conf + 0.05)
        elif k == ord('-'):
            conf = max(0.05, conf - 0.05)

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
