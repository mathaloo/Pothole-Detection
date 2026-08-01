# NOTE(Matt): The function needed to be redefined here with the necessary functions
#      in order to access it for testing purposes.
#      The test should be still ran the same way as the others:
#          "py (-3.13 for Matt env) -B -m testing.inf_worker.inf_worker_ut" from project root

from ultralytics import YOLO
import time

model = YOLO("best.pt")

SKIP_FRAMES = 1
conf = 0.45
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
DEVICE = 0

stop_event = None
latest_frame = None
latest_result = None
lock = None

def inference_worker():
        local_fc = 0
        while not stop_event.is_set():
            with lock:
                # For testing
                print("locking latest_frame in worker definition to set local frame")

                frame = latest_frame[0]
            if frame is None:
                time.sleep(0.001)
                continue
            local_fc += 1
            if local_fc % max(1, SKIP_FRAMES) != 0:
                time.sleep(0.001)
                continue
            # added for tracking inference duration during testing
            start = time.perf_counter()

            result = model.predict(
                source=frame,
                conf=conf,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
                half=True,
            )[0]
            # added both lines for tracking inference duration during testing
            end = time.perf_counter()
            print(f"Inference time: {(end - start) * 1000:.2f} ms")

            with lock:
                # For testing
                print("locking latest_result in worker definition after inference")

                latest_result[0] = (frame, result)