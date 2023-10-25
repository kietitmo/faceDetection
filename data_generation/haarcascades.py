import cv2
import os

# Sử dụng Cascade Classifier để nhận dạng khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Tạo thư mục để lưu ảnh khuôn mặt
if not os.path.exists('faces'):
    os.makedirs('faces')

count = 0  # Số lượng ảnh khuôn mặt đã được lưu
while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Chuyển đổi ảnh sang ảnh xám (để tăng hiệu suất)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận dạng khuôn mặt trong khung hình
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Lưu các ảnh khuôn mặt vào thư mục 'faces'
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = frame[y:y + h, x:x + w]
        face_filename = os.path.join('faces', f'face_{count}.jpg')
        cv2.imwrite(face_filename, face)
        count += 1

    # Hiển thị video với khuôn mặt được nhận dạng
    cv2.imshow('Face Detection', frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
