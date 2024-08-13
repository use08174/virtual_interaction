import warnings
import cv2
import random
import time
from fly import fly
from cvzone.HandTrackingModule import HandDetector

height = 720
width = 1280

number = 5
point = 0
game_duration = 30  # 게임의 총 시간 (초 단위)
countdown_duration = 3  # 카운트다운 시간 (초 단위)
start_time = 0  # 게임 시작 시간을 나중에 설정

flies = []

def spon():
    flies.append(fly(random.randint(0, width), random.randint(0, height), random.randint(1, 11), random.randint(30, 70), 71, 2, "assets/prome1.png"))
    

def register():
    for i in range(number):
        spon()

def runGame(img):
    global point
    for i in flies:
        img = i.move_straight(img, random.randint(0, width), random.randint(0, height))

    # 남은 시간 계산
    remaining_time = max(0, game_duration - int(time.time() - start_time))
    timer_text = f"Time: {remaining_time}s"
    cv2.putText(img, f"Point: {point}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, timer_text, (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

def findIf(img, arr):
    global point

    for i in flies[:]:
        if i.x < arr[9][0] < i.end_x and i.y < arr[9][1] < i.end_y:
            point += i.point
            flies.remove(i)
            if random.random() < 0.15:
                spon()
            spon()
    return img

def cam():
    global start_time, point
    warnings.filterwarnings('ignore', 'SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.')
    
    detector = HandDetector(detectionCon=0.7, maxHands=2)

    cap = cv2.VideoCapture(1)
    cap.set(3, width)
    cap.set(4, height)

    # 카운트다운 표시
    for i in range(countdown_duration, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue
        frame = cv2.flip(frame, 1)

        countdown_text = f"Starting in {i}s"
        cv2.putText(frame, countdown_text, (width // 2 - 200, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow('Hand Detection Game', frame)
        cv2.waitKey(1000)  # 1초 대기

    start_time = time.time()  # 게임 시작 시간 설정
    register()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)

        hands, frame = detector.findHands(frame, draw = False)

        frame = runGame(frame)

        if hands:
            frame = findIf(frame, hands[0]['lmList'])

        cv2.imshow('Hand Detection Game', frame)

        # 게임 시간이 종료되면 루프를 탈출
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Hand Detection Game', cv2.WND_PROP_VISIBLE) < 1 or time.time() - start_time >= game_duration:
            break

    # 최종 점수를 표시
    ret, frame = cap.read()
    if ret:
        final_score_text = f"Final Score: {point}"
        cv2.putText(frame, final_score_text, (width // 2 - 300, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        cv2.imshow('Hand Detection Game', frame)
        cv2.waitKey(5000)  # 5초 동안 점수 화면 유지

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam()
    # obj = fly(10,10,10, sumin=1)
    # cv2.imshow("word", obj.skin)
    # cv2.waitKey(5000)
