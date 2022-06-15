import os.path
import cv2
import numpy as np
from PIL import Image

NUM = 0
names = dict()

def get_name(path):
    return path.split(".")[0] + "_gray.jpg"


def to_gray_file(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray img", gray_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("gray img")
    name = get_name(path)
    cv2.imwrite(name, gray_img)


def detect_face(img):
    cv2.imshow("face", img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier("resources/weight/haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray_img, 1.1, 10, 0, (80, 80), (500, 500))
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv2.imshow("face", img)
    cv2.waitKey(0)


def video_detect(video):
    cap = cv2.VideoCapture(video)
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        detect_face(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def add_face(img, name):
    global NUM
    path = "resources/faces"
    face_path = [os.path.join(path, f) for f in os.listdir(path)]
    for ids in face_path:
        id = int(os.path.split(ids)[1].split(".")[0])
        if id >= NUM:
            NUM = id + 1

    face_detector = cv2.CascadeClassifier("resources/weight/haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(img, 1.1, 10, 0, (80, 80), (500, 500))
    if len(faces) == 1:
        names[NUM] = name
        cv2.imwrite("resources/faces/" + str(NUM) + "." + name + ".jpg", img)
        NUM += 1
        return True
    else:
        return False


def update_face():
    global names
    path = "resources/faces"
    faces = []
    ids = []
    face_path = [os.path.join(path, f) for f in os.listdir(path)]
    face_detector = cv2.CascadeClassifier("resources/weight/haarcascade_frontalface_default.xml")
    for img in face_path:
        PIL_img = Image.open(img).convert("L")
        img_numpy = np.array(PIL_img, "uint8")
        face = face_detector.detectMultiScale(img_numpy, 1.1, 10, 0, (80, 80), (500, 500))
        id = int(os.path.split(img)[1].split(".")[0])
        name = os.path.split(img)[1].split(".")[1]
        for x, y, w, h in face:
            ids.append(id)
            names[id] = name
            faces.append(img_numpy[y:y + h, x:x + w])
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write("resources/weight/faces.yml")


def recognize(video):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("resources/weight/faces.yml")
    face_detector = cv2.CascadeClassifier("resources/weight/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video)
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_detector.detectMultiScale(frame, 1.1, 10, 0, (80, 80), (500, 500))
        for x, y, w, h in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            id, confidence = recognizer.predict(frame[y:y + h, x:x + w, 0])
            if confidence > 80:
                cv2.putText(frame, "unknown", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                name = names[id]
                cv2.putText(frame, "name", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def rec_frame(frame):
    global names
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("resources/weight/faces.yml")
    face_detector = cv2.CascadeClassifier("resources/weight/haarcascade_frontalface_default.xml")

    face = face_detector.detectMultiScale(frame, 1.1, 10, 0, (80, 80), (500, 500))
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        id, confidence = recognizer.predict(frame[y:y + h, x:x + w, 0])
        if confidence > 80:
            cv2.putText(frame, "unknown", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            name = names[id]
            cv2.putText(frame, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return frame


if __name__ == "__main__":
    path = "resources/imgs/luo.jpg"
    path1 = "../resources/imgs/more_face.jpg"
    video_path = "resources/videos/face.mp4"
    detect_face(img)
    # video_detect(video_path)
    recognize(video_path)
