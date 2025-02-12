import cv2
import mediapipe as mp
import numpy as np

# Mediapipeのセットアップ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# カメラを起動
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファを最小に設定
# 画像の幅と高さを設定
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 手の動きを処理するためのインスタンスを作成
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # 検出する手の最大数
    model_complexity=0,  # モデルの複雑さ
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    prev_position = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("カメラから映像を取得できませんでした。")
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # 画像を反転して左右を調整
        frame = cv2.flip(frame, 1)

        # BGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手の検出を実行
        result = hands.process(rgb_frame)

        # 検出結果に基づいて描画
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 手のランドマークを描画
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 中心座標を計算
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                print(f'x_coords: {x_coords}')
                print(f'y_coords: {y_coords}')
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)

                # 前フレームとの位置の差を計算
                if prev_position:
                    dx = center_x - prev_position[0]
                    dy = center_y - prev_position[1]

                    # 動きの方向を判定
                    direction = "Still"
                    threshold = 0.02  # 小さい動きは無視
                    if abs(dx) > abs(dy):
                        if dx > threshold:
                            direction = "Right"
                        elif dx < -threshold:
                            direction = "Left"
                    else:
                        if dy > threshold:
                            direction = "Down"
                        elif dy < -threshold:
                            direction = "Up"

                    # 動きを表示
                    cv2.putText(frame, f"movement: {direction}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # 現在の位置を更新
                prev_position = (center_x, center_y)

        # 結果を表示
        cv2.imshow('Hand Movement Detection', frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
