import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
obama_image = cv2.imread("Peoples/obama.png")
biden_image = cv2.imread("Peoples/joe.png")
lavish_image = cv2.imread("Peoples/lavish.jpeg")
# lavish_image = cv2.imread("Peoples/lavish.jpeg")

gray_obama = cv2.cvtColor(obama_image, cv2.COLOR_BGR2GRAY)
gray_biden = cv2.cvtColor(biden_image, cv2.COLOR_BGR2GRAY)
gray_lavish = cv2.cvtColor(lavish_image, cv2.COLOR_BGR2GRAY)

faces_obama = face_cascade.detectMultiScale(gray_obama, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
faces_biden = face_cascade.detectMultiScale(gray_biden, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
faces_lavish = face_cascade.detectMultiScale(gray_lavish, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


face_info = {}
for (x, y, w, h) in faces_obama:
    face_info[(x, y, w, h)] = {"name": "Barack Obama", "age": 59, "sex": "Male", "description": "Former President of the United States", "link": "https://en.wikipedia.org/wiki/Barack_Obama"}
for (x, y, w, h) in faces_biden:
    face_info[(x, y, w, h)] = {"name": "Joe Biden", "age": 78, "sex": "Male", "description": "President of the United States", "link": "https://en.wikipedia.org/wiki/Joe_Biden"}
for (x, y, w, h) in faces_lavish:
    face_info[(x, y, w, h)] = {"name": "Lavish", "age": 20, "sex": "Male", "description": "President of the United States", "link": "https://en.wikipedia.org/wiki/Joe_Biden"}


img_new = cv2.imread('Peoples/lavish.jpeg')
gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
faces_new = face_cascade.detectMultiScale(gray_new, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
recognized_faces = []
for (x, y, w, h) in faces_new:
    if (x, y, w, h) in face_info:
        info = face_info[(x, y, w, h)]
        recognized_faces.append(info["name"])
        cv2.rectangle(img_new, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_new, info["name"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
        text = f'Age: {info["age"]} Sex: {info["sex"]} Description: {info["description"]}'
        cv2.putText(img_new, text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (1000, 1000, 1000), 2)
    else:
        recognized_faces.append("Unknown")
        cv2.rectangle(img_new, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_new, "Unknown", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

cv2.imshow('Face Recognition', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Recognized faces in the new image:", recognized_faces)