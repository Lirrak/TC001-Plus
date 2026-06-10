import cv2

def dump_device_1():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open device 1 with DSHOW")
        return
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imwrite("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_dev1.png", frame)
            print("Saved test_dev1.png, shape:", frame.shape)
            break
    cap.release()

dump_device_1()
