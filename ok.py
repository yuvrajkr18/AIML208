import cv2
import os
from datetime import datetime

path = "images"
attendance_file = "Attendance.csv"


known_faces = {}
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

for file in os.listdir(path):
    img_path = os.path.join(path, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {file}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print(f"No face in {file}")
        continue

    (x, y, w, h) = faces[0]
    face_img = gray[y:y+h, x:x+w]

    name = os.path.splitext(file)[0]
    known_faces[name] = face_img

print("✅ Faces loaded:", list(known_faces.keys()))

# ================= ATTENDANCE =================
def markAttendance(name):
    with open(attendance_file, "a+") as f:
        f.seek(0)
        data = f.readlines()
        names = [line.split(",")[0] for line in data]

        if name not in names:
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            f.write(f"{name},{time}\n")

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        print("Camera error")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        matched_name = "UNKNOWN"

        for name, known_face in known_faces.items():
            try:
                # Resize both images to same size
                face_resized = cv2.resize(face_img, (100, 100))
                known_resized = cv2.resize(known_face, (100, 100))

                diff = cv2.absdiff(face_resized, known_resized)
                score = diff.mean()

                if score < 50:  # lower = better match
                    matched_name = name.upper()
                    markAttendance(matched_name)
                    break

            except:
                continue

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, matched_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)

    cv2.imshow("Face Attendance (OpenCV)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()