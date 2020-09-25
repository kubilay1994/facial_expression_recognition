import cv2
import numpy as np

from time import perf_counter

from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(
    f"{cv2.haarcascades}/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture("videos/2.mp4")

model = load_model("models/fer_model")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('videos/output.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()

    if ret == False:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 1)

        img = cv2.resize(grayscale[y:y+h, x:x+w], (48, 48))
        img.resize(1, *img.shape, 1)
        img = img / 255

        p = model.predict(img)

        y0, dy = 150, 20
        for i in range(len(emotions)):
            y = y0 + i * dy
            text = f"{emotions[i]} : {p[0][i]:.3f}"
            cv2.putText(frame, text, (150, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        [index] = np.argmax(p, axis=-1)
        predicted_emotion = emotions[index]
        cv2.putText(frame, predicted_emotion, (150, y0 + 8 * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
        break            

    out.write(frame)
    cv2.imshow("window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
