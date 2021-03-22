import cv2 

img = cv2.imread('eyfel.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)

img = cv2.drawKeypoints(gray,kp,img)

cv2.imshow('new',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
