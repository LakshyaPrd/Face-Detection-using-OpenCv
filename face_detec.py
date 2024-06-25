import cv2 as cv
img =cv.imread('group 2.jpg')
cv.imshow('people',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

harr_cascade=cv.CascadeClassifier('harr_cascade.xml')

face_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
print(f'Number of faces: {len(face_rect)}')

for(x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow("Face Dectected",img)


cv.waitKey(0)