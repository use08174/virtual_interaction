import cv2
from cvzone.HandTrackingModule import HandDetector
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from rps_ai import rps_ai
from charm import charm
from fly_game import cam
import platform

def reset_camera():
    # Initialize webcam and hand detector
    cap = cv2.VideoCapture(1 if os.name == 'posix' and platform.system() == 'Darwin' else 0 if os.name == 'nt' else 'Unknown OS')
    cap.set(3, 1280)  # Reset camera width
    cap.set(4, 720)   # Reset camera height
    return cap

cap = reset_camera()

detector = HandDetector(maxHands=1)

# Define button properties
buttons = ["파리잡기 게임", "가위바위보", "참참참"]
button_positions = [(540, 200), (540, 350), (540, 500)]
button_size = (200, 100)

selected_button_index = None
button_pressed = False

# Load a Korean font
font_path = "assets/fonts/Jua-Regular.ttf"  # 경로를 실제 폰트 경로로 설정
font = ImageFont.truetype(font_path, 40)

def draw_buttons(img, buttons, button_positions, button_size, selected_index=None):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    for i, button in enumerate(buttons):
        x, y = button_positions[i]
        width, height = button_size
        
        # Draw button with a gradient
        if i == selected_index:
            color_start = (0, 150, 0)  # Darker green for selected button
            color_end = (0, 255, 0)  # Lighter green
        else:
            color_start = (0, 0, 150)  # Darker blue for other buttons
            color_end = (0, 0, 255)  # Lighter blue
        
        for j in range(height):
            gradient_color = (
                int(color_start[0] + (color_end[0] - color_start[0]) * (j / height)),
                int(color_start[1] + (color_end[1] - color_start[1]) * (j / height)),
                int(color_start[2] + (color_end[2] - color_start[2]) * (j / height))
            )
            cv2.line(img, (x, y + j), (x + width, y + j), gradient_color, 1)
        
        # Draw button text
        text_bbox = draw.textbbox((0, 0), button, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (width - text_width) // 2
        text_y = y + (height - text_height) // 2
        draw.text((text_x, text_y), button, font=font, fill=(255, 255, 255))
    
    img = np.array(pil_img)
    return img

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image")
        cap = reset_camera()
        continue
    img = cv2.flip(img, 1)

    # Detect hands
    try:
        hands, img = detector.findHands(img)
    except:
        continue

    # Draw buttons
    img = draw_buttons(img, buttons, button_positions, button_size, selected_button_index)
    
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']

        # Highlight the button if hand is hovering over it
        for i, (x, y) in enumerate(button_positions):
            width, height = button_size
            if x < cx < x + width and y < cy < y + height:
                selected_button_index = i
                if fingers == [0, 0, 0, 0, 0]:  # If the user makes a fist
                    button_pressed = True
                break
            else:
                selected_button_index = None
    
    # Check if a button is pressed
    if button_pressed:
        if selected_button_index is not None:
            if selected_button_index == 0:
                # Launch Fly Game
                cam(cap)
            elif selected_button_index == 1:
                # Launch Rock-Paper-Scissors
                rps_ai(cap)  # Adjust with actual filename
            elif selected_button_index == 2:
                # Launch Hand Gesture Game
                charm(cap)
            button_pressed = False  # Reset button pressed state
            cap.set(3, 1280)  # Reset camera width
            cap.set(4, 720) 

    # Display the image
    cv2.imshow("Main Menu", img)
    
    key = cv2.waitKey(1)
    if key == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
