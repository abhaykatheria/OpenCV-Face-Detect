import cv2
#importing the library cv2 inorder to do image processing stuffs

#uploading the harcascade file
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#reading the image
img = cv2.imread("image.jpg")

#converting the image into grayscale cause viola jones algorithm works best in grayscale images to detect harr like features
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detecting harr like features 
faces = face.detectMultiScale(gray_img, scaleFactor=1.07,minNeighbors= 5)


#drawing rectangle around the face found 
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y),(x+w,y+h),(12,25,0),2)


#printing out the faces cordinate data in form of numpy n dimensional array
print(faces)
 
print(type(faces))

#showing the rectangles on the original image .. not the grayscale image 
cv2.imshow("bi",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
