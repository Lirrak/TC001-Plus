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
                yuy2 = frame.reshape((240, 320, 2))
                bgr = cv2.cvtColor(yuy2, cv2.COLOR_YUV2BGR_YUY2)
                cv2.imwrite("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_yuy2_visual.png", bgr)
                print("Saved test_yuy2_visual.png")
            break
    cap.release()

dump_info(0, cv2.CAP_MSMF, 256, 384, True, "Y16 ")
