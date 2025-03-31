import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Initialize Webcam and FaceMesh Detector
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Detect Face and Face Landmarks
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]  # Get first detected face
        
        # Extract Right and Left Eye Landmarks
        right_eye_indices = [33, 160, 158, 133, 153, 144]  # Right eye points
        left_eye_indices = [362, 385, 387, 263, 373, 380]  # Left eye points

        # Get Eye Regions
        right_eye = np.array([face[i] for i in right_eye_indices], np.int32)
        left_eye = np.array([face[i] for i in left_eye_indices], np.int32)

        # Draw Eye Contours
        cv2.polylines(img, [right_eye], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(img, [left_eye], isClosed=True, color=(0, 0, 255), thickness=1)

        # Get Bounding Box for Right Eye and Crop it
        x, y, w, h = cv2.boundingRect(right_eye)
        eye_roi = img[y:y+h, x:x+w]

        # Convert to Grayscale and Apply Threshold
        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        _, threshold_eye = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY_INV)

        # Find Contours (Detect Pupil)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)  # Get largest contour (iris)
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(eye_roi, (cx, cy), 3, (255, 0, 0), -1)  # Mark iris center

    # Show Output
    cv2.imshow("Eye Tracking", img)

    # Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
