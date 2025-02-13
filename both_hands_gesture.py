import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 両手を検出
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# 指の形状を判定する関数
def recognize_gesture(landmarks):
    fingers = []

    # 親指の判定
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y  # 親指が上向き
    thumb_open = abs(thumb_tip.x - thumb_mcp.x) > 0.1  # 親指が開いている

    # 人差し指～小指の判定
    for tip_id in [8, 12, 16, 20]:
        pip_id = tip_id - 2
        mcp_id = tip_id - 3
        is_extended = landmarks[tip_id].y < landmarks[pip_id].y < landmarks[mcp_id].y
        fingers.append(is_extended)

    # ジェスチャー認識
    if all(fingers):
        return "Paper"  # すべての指が開いている
    elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return "Scissors"  # 人差し指と中指が開いている
    # [TODO] 👍️の処理の判定精度を上げるのが困難
    elif thumb_up and not any(fingers):
        return "Good"  # 親指が立っていて、他の指が閉じている
    elif not any(fingers):
        return "Rock"  # すべての指が閉じている

    return "Unknown Gesture"

# カメラ起動
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

    left_hand_gesture = None
    right_hand_gesture = None

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 手の向きを取得
            handedness = results.multi_handedness[idx].classification[0].label  # "Left" または "Right"

            # ランドマークを描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 指の形状を認識
            gesture = recognize_gesture(hand_landmarks.landmark)

            # 右手・左手の判定
            if handedness == "Left":
                left_hand_gesture = gesture
            else:
                right_hand_gesture = gesture

            # 片手の結果を表示
            cv2.putText(frame, f"{handedness}: {gesture}", (10, 50 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 両手が「パー（Paper）」の場合
    if left_hand_gesture == "Paper" and right_hand_gesture == "Paper":
        cv2.putText(frame, "Both Hands Paper", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # 映像を表示
    cv2.imshow('Hand Gesture Recognition', frame)

    # 'q'キーで終了
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
