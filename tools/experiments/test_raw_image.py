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
            if frame.dtype == np.uint8 and frame.size == 153600:
                data = frame.tobytes()
                vals = np.frombuffer(data, dtype="<u2").reshape((240, 320))
                
                # Normalize and save as image
                norm = cv2.normalize(vals, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite("test_thermal_153600.png", norm)
                
                print("Saved test_thermal_153600.png")
            elif frame.dtype == np.uint8 and frame.size == 115200:
                data = frame.tobytes()
                vals = np.frombuffer(data, dtype="<u2").reshape((180, 320))
                
                # Normalize and save as image
                norm = cv2.normalize(vals, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite("test_thermal_115200.png", norm)
                
                print("Saved test_thermal_115200.png")
            break
    cap.release()

dump_info(0, cv2.CAP_MSMF, 256, 384, True, "Y16 ")
dump_info(0, cv2.CAP_MSMF, 256, 192, True, "Y16 ")
