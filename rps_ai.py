import random
import cv2
import os
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time

def rps_ai(cap):
    cap.set(3, 640)
    cap.set(4, 480)
    
    detector = HandDetector(maxHands=1)
    
    timer = 0
    stateResult = False
    startGame = False
    scores = [0, 0]  # [AI, Player]
    game_over = False
    win_message = None
    
    while True:
        # try : 
            imgBG = cv2.imread("assets/BG.png")
            success, img = cap.read()
            
            img = cv2.flip(img, 1)
        
            imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
            imgScaled = imgScaled[:, 80:480]
        
            # Find Hands
            hands, img = detector.findHands(imgScaled)  # with draw
            
            # 하단 중앙에 문구 추가
            cv2.putText(imgBG, 'Show your palm for the machine to understand easier!', (180, 700), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2) 
            
            if startGame:
        
                if stateResult is False:
                    timer = time.time() - initialTime
                    cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)
        
                    if timer > 3:
                        stateResult = True
                        timer = 0
        
                        if hands:
                            playerMove = None
                            hand = hands[0]
                            fingers = detector.fingersUp(hand)
                            if fingers == [0, 0, 0, 0, 0]:
                                playerMove = 1
                            if fingers == [1, 1, 1, 1, 1]:
                                playerMove = 2
                            if fingers == [0, 1, 1, 0, 0]:
                                playerMove = 3
        
                            randomNumber = random.randint(1, 3)
                            imgAI = cv2.imread(f'assets/{randomNumber}.png', cv2.IMREAD_UNCHANGED)
                            imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))
        
                            # Player Wins
                            if (playerMove == 1 and randomNumber == 3) or \
                                    (playerMove == 2 and randomNumber == 1) or \
                                    (playerMove == 3 and randomNumber == 2):
                                scores[1] += 1
        
                            # AI Wins
                            if (playerMove == 3 and randomNumber == 1) or \
                                    (playerMove == 1 and randomNumber == 2) or \
                                    (playerMove == 2 and randomNumber == 3):
                                scores[0] += 1
        
                            if max(scores) >= 3:  # 종료 조건 예시: 한쪽이 3점 이상
                                game_over = True
                                if scores[0] > scores[1]:
                                    win_message = "You Lose!"
                                else:
                                    win_message = "You Win!"
        
            imgBG[234:654, 795:1195] = imgScaled
        
            if stateResult:
                imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))
        
            cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
            cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
        
            # OK 손 표시를 인식하여 게임 시작
            if hands:
                hand = hands[0]
                fingers = detector.fingersUp(hand)
                # OK 손 표시: 검지와 엄지만 펴져 있는 상태
                if fingers == [1, 0, 0, 0, 0]:
                    startGame = True
                    initialTime = time.time()
                    stateResult = False
        
            cv2.imshow("BG", imgBG)
        
            key = cv2.waitKey(1)
            
            if 3 in scores:  # Esc 키를 누르거나 게임 종료 조건이 충족되면
                # 검은색 배경 화면 생성
                result_screen = 255 * np.ones(shape=[480, 640, 3], dtype=np.uint8)
                if win_message:
                    cv2.putText(result_screen, win_message, (50, 240), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 6, cv2.LINE_AA)
                    cv2.putText(result_screen, 'Press ESC to exit', (100, 400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
                
                cv2.imshow("Result", result_screen)
                key = cv2.waitKey(1)
                time.sleep(3)
                break
            if key == 27 :
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    rps_ai(cap)
    cap.release()