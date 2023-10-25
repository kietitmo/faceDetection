import cv2
import numpy as np

if __name__ == "__main__":
    unknown_emb = np.ones(512)
#     cap = cv2.VideoCapture(1)
#     while True:
#         # Đọc khung hình từ webcam
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#
#         # Chuyển đổi ảnh sang ảnh xám (để tăng hiệu suất)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Hiển thị video với khuôn mặt được nhận dạng
#         cv2.imshow('Face Detection', frame)
#
#         # Thoát khỏi vòng lặp nếu nhấn phím 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Giải phóng tài nguyên
#     cap.release()
#     cv2.destroyAllWindows()
