import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

# Setup directory to save frame
save_in_dir = "D:/KQ_HK232/DATN/Backend_232/detectLanev2/input_frame" 
if not os.path.exists(save_in_dir):
    os.makedirs(save_in_dir)

save_out_dir = "D:/KQ_HK232/DATN/Backend_232/detectLanev2/output_frame" 
if not os.path.exists(save_out_dir):
    os.makedirs(save_out_dir)
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def k_means_clustering(points, k, max_iterations=100):
    # Step 1: Initialize random centers
    centers = points[np.random.choice(len(points), k, replace=False)]

    # Initialize clusters
    clusters = [[] for _ in range(k)]

    for _ in range(max_iterations):
        # Step 2: Assign objects to their nearest center
        new_clusters = [[] for _ in range(k)]
        for point in points:
            distances = [euclidean_distance(point, center) for center in centers]
            nearest_center_idx = np.argmin(distances)
            new_clusters[nearest_center_idx].append(point)

        # Step 3: Check for convergence
        if new_clusters == clusters:
            break
        clusters = new_clusters

        # Step 4: Update centers
        for i in range(k):
            centers[i] = np.mean(clusters[i], axis=0)

    return clusters, centers

def backend(video_binary, video_original):
    global last_left_theta, last_left_line 
    global last_right_theta, last_right_line
    global frame_width, frame_height
    #Variables use to read video and write image, video
    in_img_count = 0
    out_img_count = 0
    is_first_5_frame = False
    output_model = cv2.VideoCapture(video_binary) # Binary output video of model 
    original = cv2.VideoCapture(video_original) # Original video
    frame_width = int(output_model.get(3))
    frame_height = int(output_model.get(4))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    if not output_model.isOpened():
        print("Cannot open output video of model.")
        return
    if not original.isOpened():
        print("Cannot open original video.")
        return 

    # Array of frames, right lines and left lines
    frame_count = 0
    # Danh sách để lưu trữ tọa độ của các đường thẳng trong hộp
    # points_per_box = []
    # points_in_boxes = []
    # error_frame = 0
    while True:
        ret_model, frame_model = output_model.read()
        ret_ori, frame_ori = original.read()
        if not ret_model or not ret_ori: 
            # Can't read frame from binary output video of model and original video
            break
        # If frame can read, push frame of original to frame_array
        # frame_array.append(frame_ori)
        frame_count = frame_count + 1
        image_name = f"image_{in_img_count}.jpg"
        image_path = os.path.join(save_in_dir, image_name)
        cv2.imwrite(image_path, frame_model)
        in_img_count += 1
        
        
        # Reduce noise
        gray_frame = cv2.cvtColor(frame_model, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        kernel = np.ones((3, 3), np.uint8) 
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2) 
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # frame_model_arr.append(closing)
        lines = cv2.HoughLinesP(closing, 1, np.pi / 180, threshold=50, minLineLength=35, maxLineGap=5)
    # NGUYEN   
    #QUYNH
        points_in_boxes = []
        for contour in contours:    # Lặp qua từng contour để tìm hộp bao quanh
            rect = cv2.minAreaRect(contour) # Tính toán hộp bao quanh contour
            box = cv2.boxPoints(rect)
            box = np.int_(box)
            area = cv2.contourArea(box)          # Tính diện tích của hộp
            if area > 300:                      # Kiểm tra diện tích của hộp
                # cv2.drawContours(frame_model, [box], 0, (0, 255, 0), 2) # Vẽ hộp bao quanh contour
                points_per_box = []
                # Sử dụng phép biến đổi Hough để phát hiện các đường thẳng trong hộp               
                if lines is not None:
                    # Lặp qua các đường thẳng tìm thấy
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # Lưu tọa độ của các điểm đầu và cuối của đường thẳng vào danh sách
                        # points_per_box.append(((x1, y1), (x2, y2)))
                        points_per_box.append((x1, y1))
                        points_per_box.append((x2, y2))
                        # Vẽ đường thẳng lên ảnh gốc
                        cv2.line(frame_model, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                    # Thêm danh sách các điểm trong hộp vào danh sách tổng
                    points_in_boxes.append(points_per_box)

            # In ra tọa độ của các đường thẳng trong từng hộp
            centroids = 5
            for i, point in enumerate(points_in_boxes):
                print(f"Box {i+1}:")
                print(len(points_in_boxes))
                
                # if len(points_in_boxes) > 2:
                #     k = 2
                #     clusters, centers = k_means_clustering(points_in_boxes, k)
                #     print("Clusters:")
                #     for i, cluster in enumerate(clusters):
                #         print(f"Cluster {i + 1}: {cluster}")
                #     print("Centers:")
                #     print(centers)


        out.write(frame_model)
        image_name = f"image_{out_img_count}.jpg"
        image_path = os.path.join(save_out_dir, image_name)
        cv2.imwrite(image_path, frame_model)
        out_img_count += 1
        cv2.imshow('Processed Frame', frame_model)
        
        if cv2.waitKey(1) == ord('q'):
            break   
    output_model.release()
    original.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    backend("simulation_detect.mp4", "simulation_original.mp4")

                
if __name__ == '__main__':
    main()
