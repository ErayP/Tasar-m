import cv2
def read_and_increment():
    try:

        with open("person.txt", "r") as file:
            content = file.read().strip()

            if content:
                kisi = int(content)
            else:
                kisi = 0
    except FileNotFoundError:
        kisi = 0

    kisi += 1

    with open("person.txt", "w") as file:
        file.write(str(kisi))

    return kisi

cap = cv2.VideoCapture(0)
faceDedector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceName = read_and_increment()
faceNumber = 1

while(True):
    succes,img = cap.read()
    if not succes:
        break
    imgGRY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDedector.detectMultiScale(imgGRY,1.3,5)

    for (x,y,h,w) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imwrite("Faces/User"+str(faceName)+"_"+str(faceNumber)+".jpg",imgGRY[y:y+h,x:x+w])
        faceNumber += 1
        cv2.imshow("Yuz Tanımlama",img)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
    elif faceNumber>250:
        break
cap.release()
cv2.destroyAllWindows()
