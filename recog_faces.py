import cv2
import face_recognition
import os

known_encodings = []
known_names = []
dataset_path = "dataset"

for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)
    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        known_encodings.append(encodings[0])
        name = os.path.splitext(file)[0].split("_")[0].capitalize()
        known_names.append(name)
    else:
        print("Tidak ada wajah ditemukan di:", file)

print("Dataset wajah berhasil dimuat:", len(known_names))

# DETEK KAMERA - pakai backend default (tanpa CAP_DSHOW)
available_cameras = []

print("\nMencari kamera yang tersedia...")
for i in range(5):
    cam_test = cv2.VideoCapture(i)  # tanpa CAP_DSHOW
    if cam_test.isOpened():
        ret, test_frame = cam_test.read()
        if ret and test_frame is not None:
            print(f"  [OK] Kamera index {i} - resolusi: {int(cam_test.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cam_test.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            available_cameras.append(i)
        else:
            print(f"  [SKIP] Index {i} terbuka tapi tidak bisa baca frame")
        cam_test.release()

if len(available_cameras) == 0:
    print("Tidak ada kamera ditemukan")
    exit()

print(f"\nKamera tersedia: {available_cameras}")
camera_index = int(input("Pilih index kamera yang ingin digunakan: "))

if camera_index not in available_cameras:
    print(f"Index {camera_index} tidak valid, menggunakan kamera pertama ({available_cameras[0]})")
    camera_index = available_cameras[0]

#bukain kamera tanpa maksa codec apapun
cam = cv2.VideoCapture(camera_index)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#debuginfo kamera yang dah di pilih
fourcc = int(cam.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
print(f"\nKamera {camera_index} aktif")
print(f"  Format : {codec}")
print(f"  Resolusi: {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# pemansan dlu : buangin beberapa frame awal
print("Warming up kamera...")
for _ in range(10):
    cam.read()

print("Memulai face recognition... (tekan ESC untuk keluar)\n")

while True:
    ret, frame = cam.read()

    if not ret or frame is None or frame.size == 0:
        print("Kamera tidak terbaca, mencoba lagi...")
        continue

    # validasi dulu tuk frame tidak hijau solid
    mean_val = cv2.mean(frame)
    if mean_val[1] > 200 and mean_val[0] < 30 and mean_val[2] < 30:
        continue

    # memperkecil frame untuk mempercepat face_recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings_list = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_list):
        # balikin koordinat ke ukuran frame asli (karena di-resize 0.5x)
        top, right, bottom, left = top*2, right*2, bottom*2, left*2

        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition AI", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()