import cv2
from cvzone.HandTrackingModule import HandDetector
import subprocess
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Initialize webcam and hand detector
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
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
    img = cv2.flip(img, 1)
    
    # Detect hands
    hands, img = detector.findHands(img)
    
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
                subprocess.run(["python", "fly_game.py"])  # Adjust with actual filename
            elif selected_button_index == 1:
                # Launch Rock-Paper-Scissors
                subprocess.run(["python", "rps_ai.py"])  # Adjust with actual filename
            elif selected_button_index == 2:
                # Launch Hand Gesture Game
                subprocess.run(["python", "charm.py"])  # Adjust with actual filename
            button_pressed = False  # Reset button pressed state

    # Display the image
    cv2.imshow("Main Menu", img)
    
    key = cv2.waitKey(1)
    if key == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
