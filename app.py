from flask import Flask, render_template, Response
import cv2
import numpy as np
import winsound
from tensorflow import keras

new_model = keras.models.load_model('my_model.h5')
app = Flask(__name__)
camera = cv2.VideoCapture(0)
frequency = 600  
duration = 1000  # Duration of 1000 ms (1 second)

def generate_frames():
    counter = 0  # Initialize the counter variable
    while True:
        cbs = 0
        success, frame = camera.read()
        if not success:
            break
        else:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            faces = face_cascade.detectMultiScale(frame, 1.1, 7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for (x, y, w, h) in faces:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (192, 192, 192), -1)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

                if len(eyes) == 0:
                    print('Eyes not detected')
                    cbs = 0
                    break

                for (ex, ey, ew, eh) in eyes:
                    cbs = 1
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]

            if cbs == 1:
                final_image = cv2.resize(eyes_roi, (224, 224))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image / 255.0

                predictions = new_model.predict(final_image)
                if predictions[0][0] < -0.2:
                    status = "Open Eyes"
                    x1, y1, w1, h1 = 0, 0, 130, 50
                    cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0, 0, 0), -1)
                    cv2.putText(frame, 'Active', (x1+int(w1/10), y1+int(h1/3)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    counter += 1
                    status = "Closed Eyes"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    if counter > 1:
                        x1, y1, w1, h1 = 0, 0, 130, 50
                        cv2.rectangle(frame, (x1, x1), (x1+w1, y1+h1), (0, 0, 0), -1)
                        cv2.putText(frame, 'Sleep Alert!!', (x1+int(w1/10), y1+int(h1/3)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        winsound.Beep(frequency, duration)
                        counter = 0

            else:
                status = prev_status

            prev_status = status

            cv2.putText(frame, status, (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
