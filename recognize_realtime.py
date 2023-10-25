import face_recognition
import cv2
import os
import numpy as np
from data_embedding.embedding import Image_Embedder
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import os

embedder = FaceNet()
encoder = LabelEncoder()

class Real_time_recognition:
    def __init__(self, model, directory):
        self.cap = cv2.VideoCapture(1)
        self.model = model
        self.name_list = os.listdir(directory)
        self.name_list.append("Unknown")

    def run(self):
        font = cv2.FONT_HERSHEY_SIMPLEX  # Chọn font chữ
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)

            for face_location in face_locations:
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (160, 160))
                face_image = np.expand_dims(face_image, axis=0)
                face_emb = embedder.embeddings(face_image)
                face_name = self.model.predict_person(face_emb)[0]
                proba = self.model.predict_person(face_emb)[1]
                final_name = self.name_list[int(face_name)]
                print(self.name_list)

                if (proba[0][face_name] > 0.65):
                    cv2.putText(frame, f'{face_name}_{final_name} {proba[0][face_name]*100}%', (left, top - 10), font, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f'Unknown person', (left, top - 10), font, 0.5, (0, 255, 0), 2)

            # Hiển thị video với khuôn mặt đã phát hiện
            cv2.imshow('Face Detection', frame)

            # Thoát khỏi vòng lặp nếu nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Giải phóng tài nguyên
        self.cap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     face_capture = FaceCapture()
#     person_name = input("Nhập tên của người (không dấu và không khoảng trắng): ")
#     face_capture.capture_faces(person_name)
