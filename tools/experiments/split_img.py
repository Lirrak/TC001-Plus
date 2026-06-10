import cv2

img = cv2.imread("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_dev1.png")
if img is not None:
    left = img[:, :320]
    right = img[:, 320:]
    top = img[:240, :]
    bottom = img[240:, :]
    
    cv2.imwrite("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_left.png", left)
    cv2.imwrite("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_right.png", right)
    cv2.imwrite("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_top.png", top)
    cv2.imwrite("C:/Users/Lirrak/.gemini/antigravity-ide/brain/41d14fd6-412d-40d2-bdae-e67838700797/test_bottom.png", bottom)
