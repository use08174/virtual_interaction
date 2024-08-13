import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os
import time
import subprocess

# Initialize webcam video capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2
cap.set(3, width)  # Set width to camera width
cap.set(4, height)  # Set height to camera height
cap.set(10, 150)  # Set brightness

# Initialize HandDetector from cvzone
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize variables for calculating FPS
pasttime = 0

# Initialize pen color
pen_color = (0, 0, 255)  # Default color (red)

xp, yp = 0, 0
drawing_mode = False  # Start with selection mode

img_width, img_height = 360, 455
# Create a blank canvas to draw on with size
canvas_width, canvas_height = width, height
canvas = np.zeros((canvas_height, canvas_width, 3), np.uint8)
print(width, height)
# Define the drawing area (center rectangle on the canvas)
rect_x1, rect_y1 = canvas_width / 2 - img_width / 2, canvas_height / 2 - img_height / 2
rect_x2, rect_y2 = canvas_width / 2 + img_width / 2, canvas_height / 2 + img_height / 2
print(rect_x1, rect_y1, rect_x2, rect_y2)
# Button coordinates (scaled to the canvas size)
save_button = (
    int(canvas_width * 0.8),
    int(canvas_height * 0.05),
    int(canvas_width * 0.95),
    int(canvas_height * 0.15),
)
clear_button = (
    int(canvas_width * 0.8),
    int(canvas_height * 0.2),
    int(canvas_width * 0.95),
    int(canvas_height * 0.3),
)

while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Detect hands and get landmark positions
    hands, img = detector.findHands(frame)
    if hands:
        # Get the landmarks of the first hand detected
        lmList = hands[0]["lmList"]

        # Get the tip of the index finger (landmark 8) and middle finger (landmark 12)
        x1, y1 = lmList[8][:2]
        x2, y2 = lmList[12][:2]

        # Calculate the distance between the index and middle finger tips
        distance, _, _ = detector.findDistance((x1, y1), (x2, y2))

        # Determine if the user is in drawing mode or selection mode
        if distance < 40:  # Fingers are close together, enter drawing mode
            drawing_mode = True
            # Highlight the index finger in red
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        else:
            drawing_mode = False

        # Adjust the x-coordinates for the canvas to match the flipped frame
        canvas_x1 = int((x1 - rect_x1) * canvas_width / (rect_x2 - rect_x1))
        canvas_y1 = int((y1 - rect_y1) * canvas_height / (rect_y2 - rect_y1))

        if (
            drawing_mode
            and 0 <= canvas_x1 <= canvas_width
            and 0 <= canvas_y1 <= canvas_height
        ):
            if xp == 0 and yp == 0:
                xp, yp = canvas_x1, canvas_y1

            # Draw lines on the canvas when in drawing mode
            cv2.line(canvas, (xp, yp), (canvas_x1, canvas_y1), pen_color, 10)
            xp, yp = canvas_x1, canvas_y1
        else:
            xp, yp = 0, 0  # Reset if outside the drawing area

        # Extract x, y coordinates of the thumb and index finger tips
        thumb_tip = tuple(lmList[4][:2])
        index_tip = tuple(lmList[8][:2])

        # Check if the thumb and index finger are close together to simulate a button click
        if detector.findDistance(thumb_tip, index_tip)[0] < 40:
            if (
                save_button[0] < x1 < save_button[2]
                and save_button[1] < y1 < save_button[3]
            ):
                cv2.imwrite("canvas.jpg", canvas)  # Save the canvas as is (723x886)
                print("Canvas saved")

            if (
                clear_button[0] < x1 < clear_button[2]
                and clear_button[1] < y1 < clear_button[3]
            ):
                canvas = np.zeros(
                    (canvas_height, canvas_width, 3), np.uint8
                )  # Clear the canvas
                print("Canvas cleared")

    cv2.rectangle(frame, save_button[:2], save_button[2:], (0, 255, 0), cv2.FILLED)
    cv2.putText(
        frame,
        "Save",
        (save_button[0] + 10, save_button[1] + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.rectangle(frame, clear_button[:2], clear_button[2:], (0, 0, 255), cv2.FILLED)
    cv2.putText(
        frame,
        "Clear",
        (clear_button[0] + 10, clear_button[1] + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # Draw the rectangle for the drawing area on the frame
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), 2)

    # Show the webcam frame
    cv2.imshow("Webcam and Canvas", frame)

    # Show the canvas in a separate window
    cv2.imshow("canvas", canvas)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
