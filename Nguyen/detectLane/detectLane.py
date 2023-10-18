import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread('image/image/30.png')

# Chuyển đổi sang không gian màu HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Tạo mask cho màu xanh và màu trắng trong không gian màu HSV
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 25, 255])
white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

# Kết hợp mask lại với nhau
combined_mask = cv2.bitwise_or(blue_mask, white_mask)

# Tìm contour trong mask kết hợp
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tạo một hằng số cho chu vi tối thiểu
min_contour_length = 100  # Điều chỉnh giá trị theo nhu cầu của bạn

# Lọc ra các contour có chu vi lớn hơn hằng số
filtered_contours = [contour for contour in contours if cv2.arcLength(contour, closed=True) > min_contour_length]

# Vẽ đường viền lên ảnh gốc
cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)
# image = cv2.resize(image, (0,0), fx = 2.5, fy = 2.5)
# Hiển thị ảnh kết quả
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
