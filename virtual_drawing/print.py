import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import requests
from io import BytesIO
import time

# Define the styles
styles = {
    "Picasso": "transfer to Picasso painting style.",
    "Cartoon": "transfer to cartoon sketch style.",
    "Watercolor": "transfer to watercolor texture.",
    "Realistic": "transfer to realistic style.",
    "Surrealist": "transfer to surrealist style.",
    "Impressionist": "transfer to impressionist style.",
    "Minimalist": "transfer to minimalist style.",
    "Expressionist": "transfer to expressionist style.",
    "ArtDeco": "transfer to art deco style.",
}

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Load the canvas image
canvas_img = cv2.imread("./canvas.jpg")

# Load the frame image
frame_img = cv2.imread("./frame.png")

# Button coordinates and sizes for 3x3 grid
button_width = 150
button_height = 50
grid_size = (3, 3)  # 3x3 grid
grid_spacing = 20  # Spacing between buttons
start_x, start_y = 50, 50  # Start position for the grid

buttons = [
    {
        "name": style,
        "coords": (
            start_x + (i % grid_size[0]) * (button_width + grid_spacing),
            start_y + (i // grid_size[1]) * (button_height + grid_spacing),
        ),
    }
    for i, style in enumerate(styles.keys())
]


# Function to resize and place the styled image within the frame
def place_in_frame(styled_img_path):
    styled_img = Image.open(styled_img_path).convert("RGB")
    final_width, final_height = 803, 1196
    image_width, image_height = 723, 886
    top_padding = 30
    side_padding = 40

    # Resize the styled image to fit within the frame
    styled_img = styled_img.resize(
        (image_width, image_height), Image.Resampling.LANCZOS
    )

    # Open the frame image
    frame_img = Image.open("frame.png").convert("RGB")

    # Create a new blank image with the same size as the frame
    combined_img = Image.new("RGB", (final_width, final_height))

    # Paste the frame onto the blank image
    combined_img.paste(frame_img, (0, 0))

    # Paste the styled image onto the combined image at the correct position
    combined_img.paste(styled_img, (side_padding, top_padding))

    # Save or display the combined image
    combined_image_path = styled_img_path.replace(".png", "_framed.png")
    combined_img.save(combined_image_path)

    # Display the combined image using OpenCV
    combined_img_cv = cv2.imread(combined_image_path)
    cv2.imshow("Styled Image with Frame", combined_img_cv)


# Function to apply style to the image
def apply_style(style_name):
    prompt = styles[style_name]
    print(f"Applying style: {style_name}")

    # Use actual frame dimensions
    frame_height, frame_width = frame.shape[:2]

    loading_text = f"Loading {style_name} style..."
    cv2.putText(
        frame,
        loading_text,
        (int(frame_width // 2) - 100, int(frame_height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.imshow("Webcam and Canvas", frame)
    cv2.waitKey(1)

    try:
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/control/sketch",
            headers={
                "authorization": f"sk-wzLs50Z4qRIK7ocsnvgEJ4RfeyCaIB4gmWYkIZogQrAVIPyT",
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

            # Place the styled image within the frame
            place_in_frame(styled_image_path)

        else:
            raise Exception(str(response.json()))

    except Exception as e:
        print(f"Error applying style: {e}")


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Calculate the position to center the canvas image
    canvas_x = (frame.shape[1] - canvas_img.shape[1]) // 2
    canvas_y = (frame.shape[0] - canvas_img.shape[0]) // 2

    # Ensure canvas_img fits within the frame dimensions
    if (
        canvas_img.shape[0] <= frame.shape[0] - canvas_y
        and canvas_img.shape[1] <= frame.shape[1] - canvas_x
    ):
        # Draw the canvas image onto the webcam feed
        frame[
            canvas_y : canvas_y + canvas_img.shape[0],
            canvas_x : canvas_x + canvas_img.shape[1],
        ] = canvas_img

    # Draw the buttons onto the webcam feed
    for button in buttons:
        x, y = button["coords"]
        cv2.rectangle(
            frame, (x, y), (x + button_width, y + button_height), (0, 0, 0), cv2.FILLED
        )
        cv2.putText(
            frame,
            button["name"],
            (x + 10, y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        x, y = lmList[8][:2]  # Tip of the index finger

        # Detect thumb and index finger pinch (distance between tips is small)
        thumb_tip = lmList[4][:2]
        index_tip = lmList[8][:2]
        distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

        if distance < 40:  # Adjust the threshold as needed
            print(distance)
            for button in buttons:
                bx, by = button["coords"]
                if True:
                    cv2.rectangle(
                        frame,
                        (bx, by),
                        (bx + button_width, by + button_height),
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        button["name"],
                        (bx + 10, by + 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    apply_style(button["name"])
                    break

    # Display the webcam frame
    cv2.imshow("Webcam and Canvas", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
