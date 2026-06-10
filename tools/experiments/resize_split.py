import cv2

img = cv2.imread("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_dev1.png")
if img is not None:
    left = img[:, :320]
    right = img[:, 320:]
    
    left_resized = cv2.resize(left, (320, 240))
    right_resized = cv2.resize(right, (320, 240))
    
    cv2.imwrite("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_left_resized.png", left_resized)
    cv2.imwrite("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_right_resized.png", right_resized)
