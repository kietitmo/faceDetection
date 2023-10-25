import face_recognition
import cv2
import os

class FaceCapture:
    def __init__(self):
        # Khởi tạo webcam
        self.cap = cv2.VideoCapture(1)

    # def extract_face(self, filename, person_name):
    #     person_name = person_name.replace(" ", "")  # Loại bỏ khoảng trắng và dấu tiếng Việt
    #
    #     # Tạo thư mục để lưu các ảnh khuôn mặt cho người đó
    #     person_folder = os.path.join('faces', person_name)
    #
    #     if not os.path.exists(person_folder):
    #         os.makedirs(person_folder)
    #
    #     img = cv2.imread(filename)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     x, y, w, h = self.detector.detect_faces(img)[0]['box']
    #     x, y = abs(x), abs(y)
    #     face = img[y:y + h, x:x + w]
    #     face_arr = cv2.resize(face, self.target_size)
    #     return face_arr

    def capture_faces(self, person_name, capture_count=100):
        # Tạo thư mục để lưu ảnh khuôn mặt
        # Nhập tên của người từ bàn phím
        person_name = person_name.replace(" ", "")  # Loại bỏ khoảng trắng và dấu tiếng Việt

        # Tạo thư mục để lưu các ảnh khuôn mặt cho người đó
        person_folder = os.path.join('faces', person_name)

        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        font = cv2.FONT_HERSHEY_SIMPLEX  # Chọn font chữ
        count = 0  # Biến đếm để tạo tên duy nhất cho các tệp

        while count < capture_count:
            # Đọc khung hình từ webcam
            ret, frame = self.cap.read()

            if not ret:
                break

            # Chuyển đổi khung hình sang một định dạng mà face_recognition có thể sử dụng
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Sử dụng face_recognition để tìm khuôn mặt trong khung hình
            face_locations = face_recognition.face_locations(rgb_frame)

            # Vẽ khung xung quanh khuôn mặt đã tìm thấy
            for face_location in face_locations:
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Trích xuất khuôn mặt từ khung hình
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (160, 160))

                # Tạo tên tệp duy nhất cho khuôn mặt và lưu vào thư mục của người đó
                filename = os.path.join(person_folder, f'{person_name}_{len(os.listdir(person_folder))}.jpg')
                cv2.imwrite(filename, face_image)

                # Ghi tên của tệp lên màn hình
                cv2.putText(frame, f'Captured {count + 1}/{capture_count}', (10, 30), font, 1, (0, 0, 255), 2)
                count += 1

            # Hiển thị video với khuôn mặt đã phát hiện
            cv2.imshow('Face Detection', frame)

            # Thoát khỏi vòng lặp nếu nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Giải phóng tài nguyên
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_capture = FaceCapture()
    person_name = input("Nhập tên của người (không dấu và không khoảng trắng): ")
    face_capture.capture_faces(person_name)
