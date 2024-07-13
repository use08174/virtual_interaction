import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep

from pynput.keyboard import Controller
import cvzone

# button class
class Button():
    def __init__(self, pos:list, text:str, size:list=[85, 85]) -> object:
        self.pos = pos
        self.size = size
        self.text = text

        self.color = [255, 0, 255]
        self.textColor = [255, 255, 255]

        self.offsetx = 20
        self.offsety = 60

    def draw(self, img):
        x, y = self.pos
        w, h = self.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
        cv2.rectangle(img, self.pos, (x + w, y + h), self.color, cv2.FILLED)
        cv2.putText(img, self.text, (x + self.offsetx, y + self.offsety),
                    cv2.FONT_HERSHEY_PLAIN, 4, self.textColor, 4)
    
    def hover(self, img):
        x, y = self.pos
        w, h = self.size
        cv2.rectangle(img, self.pos, (x + w, y + h), self.color_inverse(), cv2.FILLED)
        cv2.putText(img, self.text, (x + self.offsetx, y + self.offsety),
                    cv2.FONT_HERSHEY_PLAIN, 4, self.textColor, 4)

    def clicked(self, img, size=5, color = (175, 0, 175)):
        x, y = self.pos
        w, h = self.size
        cv2.rectangle(img, (x - size, y - size), (x + w + size, y + h + size), color, cv2.FILLED)
        cv2.putText(img, self.text, (x + self.offsetx, y + self.offsety),
                    cv2.FONT_HERSHEY_PLAIN, 4, self.textColor, 4)

    def color_inverse(self):
        return (255 - self.color[0], 255 - self.color[1], 255 - self.color[2])

def drawAll(img, buttonList):
    for button in buttonList:
        button.draw(img)

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
        ["SPACE"]]

buttonList = []
final_text = ""
finger_point = 12
infinger_point = 8

# Initialize controoler for keyboard input
keyboard = Controller()

# Initialize the HandDetector with detection confidence as a float
detector = HandDetector(detectionCon=1)

# Turn on the camera
cap = cv2.VideoCapture(0)

# HD resolution
# Set the width (3); the height (4)
cap.set(3, 1280)
cap.set(4, 720)

# Rigister buttons
for i in range(len(keys) - 1):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100*j + 50, 100*i + 50], key))

while True:
    success, img = cap.read()
    if not success:
        break

		# find hand
    img = detector.findHands(img)
    # extract hand's position
    lmlist, bboxInfo = detector.findPosition(img)

		# display all the buttons
    drawAll(img, buttonList)

		# when there is a hand or hands
    if lmlist:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

						# if finger point is in between one of the button
            if x < lmlist[finger_point][0] < x + w and y < lmlist[finger_point][1] < y + h:
                button.clicked(img)
                
                # detect distance between landmarks in this case, point 12 and point 8
                l, _, _ = detector.findDistance(finger_point, infinger_point, img, draw=False)
                # print distance of the point
                print(l)

                # when clicked
                if l < 30:
                    keyboard.press(button.text)
                    button.hover(img)
                    
                    final_text += button.text
                    
                    #delay
                    sleep(0.15)

    # Display the image
    cv2.imshow("Webcam", img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
