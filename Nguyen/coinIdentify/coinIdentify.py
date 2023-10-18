import cv2
import numpy as np

# Mở video từ tệp tin
video_capture = cv2.VideoCapture('input_video.mp4')  # Thay 'ten_file_video.mp4' bằng tên tệp tin video của bạn

# Lấy kích thước video từ video gốc
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Xác định codec và tạo đối tượng VideoWriter
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))


def find_yellow_coins(frame):
    # Chuyển đổi frame sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Đặt ngưỡng cho màu vàng trong không gian màu HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Tìm các vùng màu vàng trong frame
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Tìm contour của các vùng màu vàng
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vẽ hình chữ nhật xung quanh các đồng xu màu vàng
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame
def main():
    while True:
        ret, frame = video_capture.read()  # Đọc frame từ video
        if not ret:
            break
        find_yellow_coins(frame)
        # Ghi frame đã được xử lý vào video mới
        out.write(frame)

        # Hiển thị video và hình chữ nhật
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()  # Giải phóng video capture
    out.release()  # Giải phóng VideoWriter

if __name__ == "__main__":
    main()
