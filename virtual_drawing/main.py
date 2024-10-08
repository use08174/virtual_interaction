import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
<<<<<<< HEAD
import subprocess, sys, os
=======
>>>>>>> 5cff601c39e99fe4fc0fa449e5cce7dbca0d7767


mode = 0  # 0: Drawing mode, 1: Selection mode
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3, width)  # Set width to camera width
cap.set(4, height)  # Set height to camera height
cap.set(10, 150)  # Set brightness

# Initialize HandDetector from cvzone
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize pen color
pen_color = (0, 0, 255)  # Default color (red)

xp, yp = 0, 0
drawing_mode = False  # Start with selection mode

# Define the drawing area (center rectangle on the canvas)
<<<<<<< HEAD
img_width, img_height = 360, 455  # Set rectangle width and height
=======
img_width, img_height = 360, 455
>>>>>>> 5cff601c39e99fe4fc0fa449e5cce7dbca0d7767
rect_x1 = int((width - img_width) / 2)
rect_y1 = int((height - img_height) / 2)
rect_x2 = rect_x1 + img_width
rect_y2 = rect_y1 + img_height

# Button coordinates (scaled to the canvas size)

# Button coordinates for save and clear buttons
button_width = 120
button_height = 50

save_button = (
<<<<<<< HEAD
    rect_x1 - button_width - 60,  # Left of the drawing area
    rect_y1 + 100,                      # Align with top of the drawing area
    rect_x1 - 60,
    rect_y1 + button_height + 100
=======
    int(width * 0.8),
    int(height * 0.05),
    int(width * 0.95),
    int(height * 0.15),
>>>>>>> 5cff601c39e99fe4fc0fa449e5cce7dbca0d7767
)

clear_button = (
<<<<<<< HEAD
    rect_x2 + 60,                 # Right of the drawing area
    rect_y1 + 100,                      # Align with top of the drawing area
    rect_x2 + button_width + 60,
    rect_y1 + button_height + 100
)


# Create a blank canvas to draw on with the same size as the frame
canvas = np.zeros((height, width, 3), np.uint8)


=======
    int(width * 0.8),
    int(height * 0.2),
    int(width * 0.95),
    int(height * 0.3),
)

# Create a blank canvas to draw on
canvas = np.zeros((height, width, 3), np.uint8)

>>>>>>> 5cff601c39e99fe4fc0fa449e5cce7dbca0d7767
while True:
    if mode == 0:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Detect hands and get landmark positions
        hands, img = detector.findHands(frame, draw=False)
        if hands:
            # Get the landmarks of the first hand detected
            lmList = hands[0]["lmList"]

            # Get the tip of the index finger (landmark 8) and middle finger (landmark 12)
            x1, y1 = lmList[8][:2]
            x2, y2 = lmList[12][:2]

            # Calculate the distance between the index and middle finger tips
            distance, _, _ = detector.findDistance((x1, y1), (x2, y2))

            # Determine if the user is in drawing mode or selection mode
            if distance < 35:  # Fingers are close together, enter drawing mode
                drawing_mode = True
                # Highlight the index finger in red
                cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            else:
                drawing_mode = False

<<<<<<< HEAD
            if drawing_mode and rect_x1 <= x1 <= rect_x2 and rect_y1 <= y1 <= rect_y2:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # Draw lines on the canvas when in drawing mode
                cv2.line(canvas, (xp, yp), (x1, y1), pen_color, 10)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0  # Reset if outside the drawing area

            # Extract x, y coordinates of the thumb and index finger tips
            thumb_tip = tuple(lmList[4][:2])
            index_tip = tuple(lmList[8][:2])
=======
        if drawing_mode and rect_x1 <= x1 <= rect_x2 and rect_y1 <= y1 <= rect_y2:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw lines on the canvas when in drawing mode
            cv2.line(canvas, (xp, yp), (x1, y1), pen_color, 10)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0  # Reset if outside the drawing area
>>>>>>> 5cff601c39e99fe4fc0fa449e5cce7dbca0d7767

            # Check if the thumb and index finger are close together to simulate a button click
            if detector.findDistance(thumb_tip, index_tip)[0] < 40:
                if (
                    save_button[0] < x1 < save_button[2]
                    and save_button[1] < y1 < save_button[3]
                ):
                    # Crop the drawing area from the canvas
                    cropped_canvas = canvas[rect_y1:rect_y2, rect_x1:rect_x2]

<<<<<<< HEAD
                    # Scale the cropped image to double its size (720x910 pixels)
                    scaled_canvas = cv2.resize(cropped_canvas, (720, 910), interpolation=cv2.INTER_LINEAR)

                    # Save the scaled image
                    cv2.imwrite("canvas.jpg", scaled_canvas)
                    print("Canvas saved")
                    # Run the print.py script using subprocess
                    venv_python = os.path.abspath("../../v/Scripts/python.exe")

                    # Specify the absolute path to your print.py script
                    script_path = os.path.abspath("./print.py")
=======
        # Check if the thumb and index finger are close together to simulate a button click
        if detector.findDistance(thumb_tip, index_tip)[0] < 40:
            if (
                save_button[0] < x1 < save_button[2]
                and save_button[1] < y1 < save_button[3]
            ):
                cv2.imwrite("canvas.jpg", canvas)  # Save the current canvas
                print("Canvas saved")

            if (
                clear_button[0] < x1 < clear_button[2]
                and clear_button[1] < y1 < clear_button[3]
            ):
                canvas = np.zeros_like(canvas)  # Clear the canvas
                print("Canvas cleared")

    # Blend the canvas with the frame
    frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Draw the buttons
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
>>>>>>> 5cff601c39e99fe4fc0fa449e5cce7dbca0d7767

                    # Debugging: Check if the paths exist
                    if not os.path.exists(venv_python):
                        print(f"Python executable not found: {venv_python}")
                    if not os.path.exists(script_path):
                        print(f"Script not found: {script_path}")

                    # Run the print.py script using subprocess
                    subprocess.Popen([venv_python, script_path])
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()  # Exit the program
                if (
                    clear_button[0] < x1 < clear_button[2]
                    and clear_button[1] < y1 < clear_button[3]
                ):
                    canvas = np.zeros_like(canvas)  # Clear the canvas
                    print("Canvas cleared")

<<<<<<< HEAD
        # Blend the canvas with the frame
        frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

        # Draw the buttons
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
=======
    # Show the webcam frame with drawing
    cv2.imshow("Webcam and Canvas", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
>>>>>>> 5cff601c39e99fe4fc0fa449e5cce7dbca0d7767

        # Draw the rectangle for the drawing area on the frame
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), 2)

        # Show the webcam frame with drawing
        cv2.imshow("Webcam and Canvas", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
