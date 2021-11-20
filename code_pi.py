import cv2 as cv

capture = cv.VideoCapture(1)


def rescale_frame(frame, scale=1):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)

    dimensions = (width, heigth)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def change_res(width, heigth):
    capture.set(3, width)
    capture.set(4, heigth)


while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    eye_cascade = cv.CascadeClassifier('haar_eye.xml')
    face_cascade = cv.CascadeClassifier('haar_face.xml')

    face_rect = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6)

    if(len(face_rect) < 1):
        frame = cv.putText(frame, 'No face detected', (5, frame.shape[0]-10), cv.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2, cv.LINE_AA)

    elif(len(face_rect) > 1):
        frame = cv.putText(frame, 'only one face...', (5, frame.shape[0]-10), cv.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2, cv.LINE_AA)

    else:
        for (fx, fy, w, h) in face_rect:
            cv.rectangle(frame, (fx, fy), (fx+w, fy+h), (0, 255, 0), 2)
            frame = cv.putText(frame, 'Face', (fx+5, fy-5), cv.FONT_HERSHEY_SIMPLEX,
                               1, (255, 255, 0), 2, cv.LINE_AA)

            roi_gray = gray[fy:fy + h, fx:fx + w]

            eye_rect = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=15)

            if(len(eye_rect) > 1):
                frame = cv.putText(frame, 'only one face...', (5, frame.shape[0]-10), cv.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255), 2, cv.LINE_AA)

            for (x, y, w, h) in eye_rect:
                cv.rectangle(roi_gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
                x = fx+x
                y = fy+y
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                frame = cv.putText(frame, 'Eye', (x+5, y-5), cv.FONT_HERSHEY_SIMPLEX,
                                   .5, (255, 255, 0), 1, cv.LINE_AA)

            cv.imshow("test", roi_gray)

    frame_resized = rescale_frame(frame)
    cv.imshow("video", frame_resized)

    if cv.waitKey(1) & 0xFF == ord('x'):
        break

capture.release()
cv.destroyAllWindows()
