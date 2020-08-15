import cv2
def add_face(name):
    classifier = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        done = False
        image = cv2.VideoCapture(0).read()[1]
        faces = classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3)
        if len(faces) > 1:
            print("ONLY {}'S FACES SHOULD BE IN THIS IMAGE".format(name.upper()))
        if len(faces) == 1:
            for (x, y, w, h) in faces:
                face = image[y:y + h, x:x + w]
                cv2.imwrite('faces/' + name + '.png', face)
                done = True
        if done:
            break