# read (x, y, length) and draw squares on images
import sys
import cv2

img = cv2.imread("tests/Lenna.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for square in sys.stdin:
    x, y, w = map(int, square.split(" "))
    cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
