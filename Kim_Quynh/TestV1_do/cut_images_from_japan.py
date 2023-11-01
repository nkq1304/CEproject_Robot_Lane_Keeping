# Importing all necessary libraries 
import cv2 
import os 

# Read the video from specified path 
cam = cv2.VideoCapture("japan_driving.mp4") 

try: 
	
	# creating a folder named data 
	if not os.path.exists('cut_images_from_japan'): 
		os.makedirs('cut_images_from_japan') 

# if not created then raise error 
except OSError: 
	print ('Error: Creating directory of data') 

# Set the time interval for saving frames (20 seconds)
time_interval = 80  # in seconds

# Initialize frame and time variables
current_frame = 0
current_time = 0  # in seconds

while True:
    ret, frame = cam.read()

    if ret:
        # Check if the current time interval has passed
        if current_time >= time_interval:
            name = f'./cut_images_from_japan/frame_{current_frame:04d}.jpg'
            print(f'Creating... {name}')

            # Write the extracted image
            cv2.imwrite(name, frame)

            # Reset the time and increase the frame counter
            current_time = 0
            current_frame += 1
        else:
            current_time += 1

    else:
        break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 
