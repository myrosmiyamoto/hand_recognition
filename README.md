# 手の動きを検出するプログラムの説明

## 1. このプログラムは何をするの？
このプログラムは **Webカメラを使って手の動きを検出し、動いている方向を表示** するものです。Pythonの **OpenCV** と **MediaPipe** を使って、手の位置を認識し、どの方向に動いているかを判断します。

## 2. 必要なライブラリ
このプログラムを実行するためには、以下のライブラリをインストールしておく必要があります。

```bash
pip install opencv-python mediapipe numpy
```

## 3. プログラムの仕組み
### (1) ライブラリの読み込み
```python
import cv2
import mediapipe as mp
import numpy as np
```
OpenCV（cv2）はカメラ映像を扱うために使用し、MediaPipe（mp）は手のランドマーク（特徴点）を検出するために使います。

### (2) カメラのセットアップ
```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```
ここでは **カメラを起動** し、映像の幅と高さを設定しています。

### (3) 手の検出を行う準備
```python
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
```
MediaPipeの **手の検出機能** を使うための準備を行います。

### (4) 手の検出と動きの分析
```python
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
```
この部分では、手のランドマークを検出する設定を行います。

- `max_num_hands=1` → **検出する手は1つ** に制限
- `min_detection_confidence=0.7` → **検出の信頼度**（70%以上で手を検出）
- `min_tracking_confidence=0.7` → **追跡の信頼度**（70%以上で手の動きを追跡）

### (5) カメラ映像の取得と手のランドマーク検出
```python
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
result = hands.process(rgb_frame)
```
- `cap.read()` でカメラから映像を取得
- `cv2.flip(frame, 1)` で **鏡のように左右を反転**
- `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` で **色空間をBGRからRGBに変換**
- `hands.process(rgb_frame)` で **手のランドマークを検出**

### (6) 手のランドマークの描画
```python
if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
```
**手の特徴点（ランドマーク）をカメラ映像に描画** します。

### (7) 手の中心座標を計算
```python
x_coords = [lm.x for lm in hand_landmarks.landmark]
y_coords = [lm.y for lm in hand_landmarks.landmark]
center_x = np.mean(x_coords)
center_y = np.mean(y_coords)
```
手の特徴点のx座標とy座標をリストに格納し、**手の中心位置を求めます**。

### (8) 手の動く方向を判定
```python
if prev_position:
    dx = center_x - prev_position[0]
    dy = center_y - prev_position[1]

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

    cv2.putText(frame, f"movement: {direction}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
```
- **前のフレームとの位置の差** を計算して、
  - **右に動いたら "Right"**
  - **左に動いたら "Left"**
  - **上に動いたら "Up"**
  - **下に動いたら "Down"**
  - **動いていなければ "Still"**

- `cv2.putText()` を使って **画面に動きの方向を表示** します。

### (9) カメラ映像を表示
```python
cv2.imshow('Hand Movement Detection', frame)
```
カメラ映像をウィンドウに表示します。

### (10) プログラムの終了処理
```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
```
'q'キーを押すと **カメラを停止し、ウィンドウを閉じます**。

## 4. 実行方法
ターミナル（またはコマンドプロンプト）で、Pythonファイルを実行します。
```bash
python ファイル名.py
```

## 5. まとめ
- **Webカメラを使って手の動きを検出**
- **手の中心を求め、前の位置と比較して動く方向を判定**
- **結果を画面にリアルタイム表示**
- **'q'キーを押すとプログラム終了**

このプログラムを応用すれば、手の動きを使って **簡単なジェスチャー操作** もできるようになります！

