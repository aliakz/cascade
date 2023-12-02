import cv2 as cv

#  using face-cascade for detect faces
img = cv.imread("faces.jpg")


face_detector = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_alt.xml")


faces = face_detector.detectMultiScale(img, 1.1, 2)


print("faces detected :",len(faces))

for x,y,w,h in faces:
    cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 1)


# using eye-cascade for detect eyes

eye_detector = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_eye.xml")



eyes =eye_detector.detectMultiScale(img, 2, 4)
for x,y,w,h in eyes:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
print("eyes  detected :",len(faces)*2)

cv.imshow( "img ",img)
cv.waitKey(0)



# using car-cascade for detect cars

img = cv.imread("car.jpg")
img=cv.resize(img,dsize=None,fx=2,fy=2,interpolation=0)

car_detector = cv.CascadeClassifier('cars .xml')
print(car_detector)


cars =car_detector.detectMultiScale(img, 1.08, 3)
for x,y,w,h in cars:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
print("cars  number :",len(cars))

cv.imshow("cars detected : ", img)
cv.waitKey(0)


cv.destroyAllWindows()




