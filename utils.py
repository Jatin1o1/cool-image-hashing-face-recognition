import cv2
def face_detect(image):
    classifier = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_locations = classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3)
    faces = []
    if len(faces_locations) > 0:
        for (x, y, w, h) in faces_locations:
            face = image[y:y + h, x:x + w]
            faces.append(face)
    return faces