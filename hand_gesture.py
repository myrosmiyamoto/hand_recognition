import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # 検出する手の最大数
        model_complexity=0,  # モデルの複雑さ
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

# 指の形状を判定する関数
def recognize_gesture(landmarks):
    # 指の各関節のy座標を比較して、開いているか閉じているかを判定
    fingers = []

    # 人差し指～小指
    for tip_id in [8, 12, 16, 20]:
        fingers.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)

    # ジェスチャー認識
    if all(fingers):  # すべての指が開いている
        return "paper"
    elif not any(fingers):  # すべての指が閉じている
        return "rock"
    elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return "Scissors"  # 人差し指と中指が開いていて、薬指と小指が閉じている
    return "nothing"

# カメラを起動
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファを最小に設定
# 画像の幅と高さを設定
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラが検出できません。")
        break

    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    # 画像を反転して左右を調整
    frame = cv2.flip(frame, 1)

    # 画像をRGBに変換
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False

    # 手のランドマークを検出
    results = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # ランドマークを描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 指の形状を認識
            gesture = recognize_gesture(hand_landmarks.landmark)
            # 結果を表示
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 映像を表示
    cv2.imshow('Hand Gesture Recognition', frame)

    # 'q'キーで終了
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
