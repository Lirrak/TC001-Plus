import cv2
import numpy as np

def dump_info(idx, backend, width, height, raw_format, fourcc):
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened(): return
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if raw_format:
        cap.set(cv2.CAP_PROP_FORMAT, -1)
    
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Read! shape={frame.shape} dtype={frame.dtype} size={frame.size}")
            if frame.dtype == np.uint8 and frame.size in [153600, 115200, 98304, 196608]:
                data = frame.tobytes()
                vals = np.frombuffer(data, dtype="<u2")
                print(f"Vals len {len(vals)}")
                print(f"Vals min/max/avg: {vals.min()} {vals.max()} {vals.mean()}")
                # Try converting to temp
                temps = vals.astype(np.float32) / 64.0 - 273.15
                print(f"Temps min/max/avg: {temps.min()} {temps.max()} {temps.mean()}")
            break
    cap.release()

print("DSHOW:")
dump_info(0, cv2.CAP_DSHOW, 256, 384, True, "Y16 ")
print("MSMF:")
dump_info(0, cv2.CAP_MSMF, 256, 384, True, "Y16 ")
dump_info(0, cv2.CAP_MSMF, 256, 192, True, "Y16 ")
