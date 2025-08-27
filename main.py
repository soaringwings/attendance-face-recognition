import face_recognition
import cv2
import numpy as np
from datetime import datetime
from datetime import date

date = str(date.today())
with open(f'{date}.csv', 'w') as file:
    file.write('name,time marked present')


# record attendance
def record_attendance(person):
    f = open(f'{date}.csv', 'r+')
    data_list = f.readlines()
    register_names = []

    for line in data_list:
        newcomer = line.split(',')
        register_names.append(newcomer[0])

    if person not in register_names:
        if person == 'Unknown' or person == '\n':
            pass
        else:
            right_now = datetime.now()
            string_right_now = right_now.strftime('%H:%M:%S')
            f.writelines(f'\n{person},{string_right_now}')


# Code adapted from a Udemy lecture:
# https://www.udemy.com/course/total-python/learn/lecture/32670190

video_capture = cv2.VideoCapture(0)

# Load images and encodings
image1 = face_recognition.load_image_file("./images/brad_pitt.jpg")
encoding1 = face_recognition.face_encodings(image1)[0]
image2 = face_recognition.load_image_file("./images/obama.jpg")
encoding2 = face_recognition.face_encodings(image2)[0]

# Create arrays with names and encodings
known_face_encodings = [
    encoding1,
    encoding2
]
known_face_names = [
    "Edward",
    "Jayred"
]

# Create variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame to 1/4 resolution and convert colorspace
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find locations of faces and loads the face encodings of the faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"

            # Compare face distances
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.4:
                name = known_face_names[best_match_index]

            face_names.append(name)
            record_attendance(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since it was initially scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

    cv2.imshow('Video', frame)

    # Press Esc to close
    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()

# Portions of this program have been adapted from a repository by Adam Geitgey which is licensed under
# the MIT license:
# https://github.com/ageitgey/face_recognition
