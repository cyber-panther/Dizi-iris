import cv2 as cv
import winsound

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 250  # Set Duration
time = 0

def set_up():
    global capture
    global eye_cascade
    global face_cascade

    capture = cv.VideoCapture(0)
    eye_cascade = cv.CascadeClassifier('haar_eye.xml')
    face_cascade = cv.CascadeClassifier('haar_face.xml')


def message(frame, message):
    frame = cv.putText(frame, message, (5, frame.shape[0]-10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2, cv.LINE_AA)


def Buzzer():
    if(time > 30):
        print(time)
        winsound.Beep(frequency, duration)


def rescale_frame(frame, scale=1.25):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)

    dimensions = (width, heigth)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def eye_tracker(roi_gray, fx, fy, frame):

    global time

    eye_rect = eye_cascade.detectMultiScale(
        roi_gray, scaleFactor=1.1, minNeighbors=15)

    for (x, y, w, h) in eye_rect:
        x = fx+x
        y = fy+y

        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame = cv.putText(frame, 'Eye', (x+5, y-5), cv.FONT_HERSHEY_SIMPLEX,
                           .5, (255, 255, 0), 1, cv.LINE_AA)

    if(len(eye_rect) < 1):
        message(frame, 'no eye open')
        time = time + 2

    elif(len(eye_rect) == 1):
        message(frame, 'One eye open')
        time = time + 1

    elif(len(eye_rect) > 1):
        message(frame, 'Both eyes open')
        time = 0

    Buzzer()


def face_tracker(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_rect = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6)

    if(len(face_rect) < 1):
        message(frame, 'No face detected')

    elif(len(face_rect) > 1):
        message(frame, 'only one face...')

    else:
        for (fx, fy, w, h) in face_rect:
            cv.rectangle(frame, (fx, fy), (fx+w, fy+h), (0, 255, 0), 2)
            frame = cv.putText(frame, 'Face', (fx+5, fy-5), cv.FONT_HERSHEY_SIMPLEX,
                               1, (255, 255, 0), 2, cv.LINE_AA)

            roi_gray = gray[fy:fy + h, fx:fx + w]

            eye_tracker(roi_gray, fx, fy, frame)


def main():

    set_up()

    while True:
        isTrue, frame = capture.read()

        face_tracker(frame)

        frame_resized = rescale_frame(frame)
        cv.imshow("video", frame_resized)

        if cv.waitKey(1) & 0xFF == ord('x'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
