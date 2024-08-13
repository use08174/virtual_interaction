import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import time
import random

# Initialize the HandDetector with detection confidence as a float


# # Turn on the camera
# cap = cv2.VideoCapture(1 if os.name == 'posix' and platform.system() == 'Darwin' else 0 if os.name == 'nt' else 'Unknown OS')  # Use 0 for built-in webcam or adjust as needed for external webcams

# HD resolution
# Set the width (3); the height (4)



def overlay_image(img, img_overlay, x, y):
    """Overlay img_overlay on top of img at (x, y)"""
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img[y1:y2, x1:x2] = img_overlay[y1o:y2o, x1o:x2o]

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points"""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def charm(cap):
    cap.set(3, 1280)
    cap.set(4, 720)

    # detector = HandDetector(detectionCon=1)
    detector = HandDetector(detectionCon=1, maxHands=2)
    # Initialize timer variables
    timer_duration = 3  # Timer duration in seconds
    result_display_duration = 10  # Result display duration in seconds
    timer_started = False
    result_displayed = False
    start_time = None
    result_start_time = None

    captured_direction = None
    target_direction = random.choice(["Right", "Center", "Left"])

    # Load direction images and resize them to a fixed size (e.g., 300x300 pixels)
    fixed_size = (300, 200)
    right_img = cv2.imread("./assets/Right.png")
    center_img = cv2.imread("./assets/Center.png")
    left_img = cv2.imread("./assets/Left.png")

    # Resize images to the fixed size
    right_img = cv2.resize(right_img, fixed_size)
    center_img = cv2.resize(center_img, fixed_size)
    left_img = cv2.resize(left_img, fixed_size)

    # Dictionary to map directions to images
    direction_images = {
        "Right": right_img,
        "Center": center_img,
        "Left": left_img
    }

    # Text properties
    text = "Start"
    text_pos = (570, 360)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_normal = 2
    font_scale_large = 3
    color_normal = (0, 0, 0)  # Black
    color_hover = (0, 0, 255)  # Red

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # 좌우 반전
        img = cv2.flip(img, 1)

        # Find hand and extract hand's position
        hands, img = detector.findHands(img, draw=True)
        print(hands)
        if hands:
            print(hands)
            lmlist = hands[0]['lmList']
        else:
            lmlist = None
        
        # Draw text and detect hover
        if not timer_started and not result_displayed:  # Only display "Start" if the timer hasn't started and result isn't displayed
            text_size = cv2.getTextSize(text, font, font_scale_normal, 2)[0]
            text_x1, text_y1 = text_pos[0], text_pos[1] - text_size[1]
            text_x2, text_y2 = text_pos[0] + text_size[0], text_pos[1]

            if lmlist:
                # Check if hand is over the text
                hand_over_text = text_x1 < lmlist[9][0] < text_x2 and text_y1 < lmlist[9][1] < text_y2
                if hand_over_text:
                    cv2.putText(img, text, text_pos, font, font_scale_large, color_hover, 3)
                    if not timer_started and not result_displayed:
                        # Check if the hand is a fist
                        fingers = detector.fingersUp()
                        fist = fingers.count(1) == 0  # All fingers are down

                        if fist:
                            timer_started = True
                            start_time = time.time()
                else:
                    cv2.putText(img, text, text_pos, font, font_scale_normal, color_normal, 2)
            else:
                cv2.putText(img, text, text_pos, font, font_scale_normal, color_normal, 2)

        if timer_started:
            elapsed_time = time.time() - start_time

            if elapsed_time < timer_duration:
                # Update the timer
                timer = int(timer_duration - elapsed_time)
                cv2.putText(img, f"{timer}", (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
                
                # Display the center image during the countdown
                img_h, img_w, _ = img.shape
                center_x = (img_w - fixed_size[0]) // 2  # Center horizontally
                overlay_image(img, center_img, center_x, 50)  # Display center_img at the top center
            else:
                if captured_direction is None:
                    if lmlist:
                        # Calculate distances for middle finger and thumb
                        middle_finger_tip = lmlist[12]
                        middle_finger_joint = lmlist[9]
                        thumb_tip = lmlist[4]
                        thumb_joint = lmlist[2]

                        # Draw circles on the middle finger and thumb
                        cv2.circle(img, (middle_finger_tip[0], middle_finger_tip[1]), 15, (0, 255, 0), cv2.FILLED)
                        cv2.circle(img, (middle_finger_joint[0], middle_finger_joint[1]), 15, (255, 0, 0), cv2.FILLED)
                        cv2.circle(img, (thumb_tip[0], thumb_tip[1]), 15, (0, 255, 255), cv2.FILLED)
                        cv2.circle(img, (thumb_joint[0], thumb_joint[1]), 15, (255, 255, 0), cv2.FILLED)

                        # Calculate distances
                        middle_finger_distance = calculate_distance(middle_finger_tip, middle_finger_joint)
                        thumb_distance = calculate_distance(thumb_tip, thumb_joint)
                        
                        # Calculate the ratio
                        distance_ratio = middle_finger_distance / thumb_distance
                        
                        # Threshold for determining direction
                        ratio_threshold = 1.2
                        
                        if distance_ratio > ratio_threshold:
                            captured_direction = "Right"
                        elif distance_ratio < 1 / ratio_threshold:
                            captured_direction = "Left"
                        else:
                            captured_direction = "Center"
                    
                    if captured_direction:
                        # Save the frame where the direction is captured
                        cv2.imwrite("captured_frame.png", img)
                
                # Display the direction on the image if it's captured
                if captured_direction:
                    result = "이겼습니다!" if captured_direction == target_direction else "졌습니다 ㅠ"
                    
                    # Display the robot's direction, user's direction, and result as text at the bottom
                    bottom_offset = 50  # Offset from the bottom of the image
                    text_y_position = img.shape[0] - bottom_offset  # Position text at the bottom
                    font_scale_small = 1.5  # Smaller font scale
                    
                    cv2.putText(img, f"Robot: {target_direction}", (50, text_y_position - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), 2)
                    cv2.putText(img, f"User: {captured_direction}", (50, text_y_position - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), 2)
                    cv2.putText(img, f"Result: {result}", (50, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), 2)
                    
                    # Display the image corresponding to the robot's direction at the top center
                    robot_img = direction_images[target_direction]
                    overlay_image(img, robot_img, center_x, 50)  # Display robot's direction image at the top center
                    
                    # Set result_displayed to True and start result timer
                    result_displayed = True
                    result_start_time = time.time()
                    timer_started = False

        if result_displayed:
            result_elapsed_time = time.time() - result_start_time
            if result_elapsed_time < result_display_duration:
                # Display the robot's direction, user's direction, and result during the result display duration at the bottom
                bottom_offset = 50  # Offset from the bottom of the image
                text_y_position = img.shape[0] - bottom_offset  # Position text at the bottom
                font_scale_small = 1.2  # Smaller font scale
                
                cv2.putText(img, f"Robot: {target_direction}", (center_x, text_y_position - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), 2)
                cv2.putText(img, f"User: {captured_direction}", (center_x, text_y_position - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), 2)
                cv2.putText(img, f"Result: {result}", (center_x, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), 2)
                
                # Display the image corresponding to the robot's direction at the top center
                robot_img = direction_images[target_direction]
                overlay_image(img, robot_img, center_x, 50)  # Display robot's direction image at the top center
            else:
                # Reset all variables to initial state
                result_displayed = False
                captured_direction = None
                target_direction = random.choice(["Right", "Center", "Left"])

        # Display the image
        cv2.imshow("Webcam", img)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    # cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(1 if os.name == 'posix' and platform.system() == 'Darwin' else 0 if os.name == 'nt' else 'Unknown OS')
    charm(cap)
    cap.release()