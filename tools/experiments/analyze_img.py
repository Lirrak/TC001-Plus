import cv2
import numpy as np

img = cv2.imread("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_dev1.png")
if img is not None:
    left = img[:, :320]
    right = img[:, 320:]
    top = img[:240, :]
    bottom = img[240:, :]
    print("Image shape:", img.shape)
    
    print("Left mean:", left.mean())
    print("Right mean:", right.mean())
    print("Top mean:", top.mean())
    print("Bottom mean:", bottom.mean())
    
    # Are left and right drastically different?
    diff_lr = np.abs(left.mean(axis=(0,1)) - right.mean(axis=(0,1)))
    print("Diff L-R color channels:", diff_lr)
