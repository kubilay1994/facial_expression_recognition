import cv2
import numpy as np

from time import perf_counter

from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(
    f"{cv2.haarcascades}/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture("videos/2.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('videos/output.avi', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

model = load_model("models/fer_model")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

fps = cap.get(cv2.CAP_PROP_FPS)

# i = 0
# predicted_emotion = emotions[0]
while True:
    ret, frame = cap.read()

    if ret == False:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

        # if i % (fps / 3) == 0:
        img = cv2.resize(grayscale[y:y+h, x:x+w], (48, 48))
        img.resize(1, *img.shape, 1)
        img = img / 255

        # start = perf_counter()
        p = model.predict(img)

        # print(perf_counter() - start)
        [index] = np.argmax(p, axis=-1)
        # predicted_emotion = emotions[index]

        cv2.putText(frame, emotions[index], (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    out.write(frame)
    cv2.imshow("window", frame)

    # i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
