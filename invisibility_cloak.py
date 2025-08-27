import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
time.sleep(2) # Allow camera to warm up

# Capture background frame
background = 0 # Initialize background
for i in range(30):
    ret , background = cap.read() # Capture background frame
background = np.flip(background, axis= 1) # Flip background frame for mirror effect

# Detect Cloak Color
while cap.isOpened():
        ret , img = cap.read() # Capture frame-by-frame
        if not ret: 
            break 
        img = np.flip(img , axis = 1) # Flip frame for mirror effect
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert BGR to HSV
        # Define range of red color in HSV  
        lower_red = np.array([0,120,70])
        upper_red = np.array([10,255,255]) 
        mask1 = cv2.inRange(hsv,lower_red,upper_red)

        lower_red2 = np.array([170,120,70])
        upper_red2 = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red2,upper_red2)

        mask = mask1 + mask2 # Combine masks

        #clean the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8)) # Remove noise
        mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations = 1)  # Dilate the mask to fill gaps

        # Create inverse mask
        mask_inv = cv2.bitwise_not(mask) 

        # Segment out the non-cloak  part
        res1 = cv2.bitwise_and(img, img, mask=mask_inv)

        # Segment baclkground where cloak is present
        res2 = cv2.bitwise_and(background, background, mask=mask)

        # Generating the final output
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0) # Combine the two results

        cv2.imshow("invisible cloak", final_output) # Display the output

        if cv2.waitKey(1) & 0xFF == ord('q'): # Break the loop on 'q' key press
            break

cap.release()
cv2.destroyAllWindows()# Release the webcam and close all OpenCV windows
