import warnings
import cv2
import os
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

def findIf(img, hand, detector):
    global point

    # 손의 정보를 가져오기
    fingers = detector.fingersUp(hand)  # hand에는 이미 'lmList'와 'type'이 포함되어 있음
    fist = fingers.count(1) == 0  # 모든 손가락이 접혀있으면 주먹

    if fist:
        for i in flies[:]:
            if i.x < hand['lmList'][9][0] < i.end_x and i.y < hand['lmList'][9][1] < i.end_y:
                point += i.point
                flies.remove(i)
                if random.random() < 0.15:
                    spon()
                spon()
    return img




def add_frame_and_text(img):
    # 화면에 노란색 테두리 추가
    color = (0, 255, 255)  # 노란색 (BGR 형식)
    thickness = 10  # 테두리 두께
    img = cv2.rectangle(img, (0, 0), (width, height), color, thickness)

    # 상단에 텍스트 추가
    text = "Catch the Fly!"
    font_scale = 2
    thickness = 3
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = text_size[1] + 20
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return img


def cam(cap):
    global start_time, point
    warnings.filterwarnings('ignore', 'SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.')
    
    detector = HandDetector(detectionCon=0.7, maxHands=2)

    # cap = cv2.VideoCapture(1 if os.name == 'posix' and platform.system() == 'Darwin' else 0 if os.name == 'nt' else 'Unknown OS')
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
        frame = add_frame_and_text(frame)  # 테두리와 텍스트 추가

        if hands:
            frame = findIf(frame, hands[0], detector)  # `hands[0]` 전체를 전달

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

    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cam(cap)
    cap.release()