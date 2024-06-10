#library import
import cv2
import mediapipe as mp
import numpy as np
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import speech_recognition as sr
from selenium.webdriver.common.by import By

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

browserIsOpen = False
cap = cv2.VideoCapture(0)
#print(cap)

def browserStart() : #fist
    driver.get("https://www.google.com")
def browserClose() :
    driver.quit()

while cap.isOpened(): #카메라가 작동중일 떄
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1) #카메라로 들어온 이미지 좌우반전
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree
            #print(angle)

            # Inference gesture
            data = np.array([angle], dtype=np.float32) #손가락 위치 정보
            ret, results, neighbours, dist = knn.findNearest(data, 3) #입력 데이터의 클래스를 예측
            #print(ret, results)
            idx = int(results[0][0]) #손 동작에 대한 인덱스 값이 담김

            # 어떤 동작인지 출력
            if idx in rps_gesture.keys(): #rps_gesture에 해당하는 손동작이 있으면
                cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                if browserIsOpen == False and idx == 0: #fist
                    browserIsOpen = True
                    browserStart()

                elif idx == 5: #검색창에 사전에 입력한 단어를 입력
                    try:
                        recognizer = sr.Recognizer()
                        search_box = driver.find_element(By.CLASS_NAME, 'gLFyf')
                        search_box.send_keys('파이썬')
                    except selenium.common.exceptions.NoSuchElementException:
                        print("웹 아직 페이지가 실행되지 않음")

                elif idx == 9:
                    try:
                        search_box = driver.find_element(By.CLASS_NAME, 'gLFyf')
                        search_box.submit()
                    except selenium.common.exceptions.NoSuchElementException:
                        print("웹 아직 페이지가 실행되지 않음")


            # Other gestures
            # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
