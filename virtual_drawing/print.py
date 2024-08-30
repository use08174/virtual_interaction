import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import requests
from io import BytesIO
import time, sys

styles = {
    "Picasso": "transfer to Picasso painting style",
    "Watercolor": "transfer to watercolor painting style.",
    "Impressionist": "transfer to impressionist style.",
    "Expressionist": "transfer to expressionist style.",
    "Robot": "transfer to 4d robot transformer style with city background.",
    "Surrealist": "transfer to surrealist style.",
    "horror": "transfer to horror white ghost painting style",
    "pixelart": "transfer to pixel painting style with rainbow color.",
    "Statue": "transfer to 3d white concrete sculpture style with black background.",
}

# Map regions to style keys
style_map = {
    (0, 0): "Picasso",  # Top-left
    (0, 1): "Watercolor",  # Top-center
    (0, 2): "Impressionist",  # Top-right
    (1, 0): "Expressionist",  # Center-left
    (1, 1): "Robot",  # Center
    (1, 2): "Surrealist",  # Center-right
    (2, 0): "horror",  # Bottom-left
    (2, 1): "pixelart",  # Bottom-center
    (2, 2): "Statue",  # Bottom-right
}


# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Double the resolution of the webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3, width)
cap.set(4, height)

# Load the canvas image
canvas_img = cv2.imread("./canvas.jpg")

# Load the frame images
frame_red = "./images/frame_red.png"
frame_black = "./images/frame_black.png"

# Button dimensions
button_width = 120
button_height = 50

# Calculate positions for the buttons
center_x, center_y = width // 2, height // 2

# Define positions for 8 directions around the center
positions = [
    (center_x - 200, center_y - 200),  # 북서
    (center_x, center_y - 200),        # 북
    (center_x + 200, center_y - 200),  # 북동
    (center_x + 200, center_y),        # 동
    (center_x + 200, center_y + 200),  # 남동
    (center_x, center_y + 200),        # 남
    (center_x - 200, center_y + 200),  # 남서
    (center_x - 200, center_y),        # 서
    (center_x, center_y)               # 중심
]

buttons = [
    {
        "name": style,
        "coords": (positions[i][0] - button_width // 2, positions[i][1] - button_height // 2),
    }
    for i, style in enumerate(styles.keys())
]

# Placeholder for frame initialization
frame = np.zeros((height, width, 3), np.uint8)

# Define grid regions (3x3 grid)
grid_size = 3
cell_width = width // grid_size
cell_height = height // grid_size

def display_transparent_overlay(image_path, frame, alpha=0.5):
    """
    실시간 웹캠 영상 위에 반투명한 이미지를 오버레이하여 전체 화면으로 출력합니다.

    Parameters:
    - image_path: 오버레이할 이미지 파일의 경로
    - frame: 현재 웹캠에서 캡처한 프레임
    - alpha: 이미지의 투명도 (0: 완전히 투명, 1: 완전히 불투명)
    """
    # load image
    print(image_path)
    img = cv2.imread(image_path)

    img = cv2.resize(img, (width, height))

    if img is None:
        print(f"Error: Unable to load image '{image_path}'")
        return frame  

    # if no alpha, add
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # set the transparency
    img[:, :, 3] = img[:, :, 3] * alpha

    # overlay the image
    overlay = cv2.addWeighted(frame, 1-alpha, img[:, :, :3], alpha, 0)
    return overlay

# Function to resize and place the styled image within the frame
def place_in_frame(styled_img_path):
    styled_img = Image.open(styled_img_path).convert("RGB")
    final_width, final_height = 800, 1200
    image_width, image_height = 720, 910
    top_padding = 40
    side_padding = 40

    # Resize the styled image to fit within the frame
    styled_img = styled_img.resize(
        (image_width, image_height), Image.Resampling.LANCZOS
    )

    # Load the red and black frame images
    for frame_color, frame_img_path in [("red", frame_red), ("black", frame_black)]:
        frame_img = Image.open(frame_img_path).convert("RGB")

        # Create a new blank image with the same size as the frame
        combined_img = Image.new("RGB", (final_width, final_height))

        # Paste the frame onto the blank image
        combined_img.paste(frame_img, (0, 0))

        # Paste the styled image onto the combined image at the correct position
        combined_img.paste(styled_img, (side_padding, top_padding))

        # Save the combined image
        combined_image_path = styled_img_path.replace(".png", f"_{frame_color}.png")
        combined_img.save(combined_image_path)

        # Display the combined image using OpenCV
        combined_img_cv = cv2.imread(combined_image_path)
        cv2.imshow(f"Styled Image with {frame_color.capitalize()} Frame", combined_img_cv)


# Image paths corresponding to each grid region
image_paths = [f"{i}.png" for i in range(9)]

# Map grid regions to image paths
image_map = {(i // 3, i % 3): image_paths[i] for i in range(9)}

## Function to apply style to the image
def apply_style(style_name):
    prompt = styles[style_name]
    print(f"Applying style: {style_name}")

    # Display the loading image instead of filling the frame with black
    loading_img = cv2.imread("./images/loading.png")

    # Resize the loading image to fit the entire frame
    loading_img = cv2.resize(loading_img, (width, height))

    # Show the loading image
    cv2.imshow("Webcam and Canvas", loading_img)
    cv2.waitKey(1)

    try:
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/control/sketch",
            headers={
                "authorization": f"todo",
                "accept": "image/*",
            },
            files={"image": open("./canvas.jpg", "rb")},
            data={"prompt": prompt, "control_strength": 0.7, "output_format": "png"},
        )

        if response.status_code == 200:
            print("API request successful")
            img_data = response.content
            img = Image.open(BytesIO(img_data))
            styled_image_path = f"./styled_{style_name}.png"
            img.save(styled_image_path)

            # Place the styled image within both frames
            place_in_frame(styled_image_path)
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()  # Exit the program

        else:
            raise Exception(str(response.json()))

    except Exception as e:
        print(f"Error applying style: {e}")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    

    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        x, y = lmList[8][:2]  # Tip of the index finger

        # Determine which grid cell the finger is in
        col = x // cell_width
        row = y // cell_height
        
        if (row, col) in image_map:
            frame = display_transparent_overlay(f"./images/{style_map[(row, col)]}.png", frame, 0.5)
                
        # Detect thumb and index finger pinch (distance between tips is small)
        thumb_tip = lmList[4][:2]
        index_tip = lmList[8][:2]
        distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

        if distance < 40:  # Adjust the threshold as needed
            if (row, col) in style_map:
                style_name = style_map[(row, col)]
                apply_style(style_name)  # Apply the style associated with the grid cell

    # Display the webcam frame with the grid
    cv2.imshow("Webcam and Canvas", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
