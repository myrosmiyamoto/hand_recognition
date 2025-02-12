# 手の動きを検出するプログラムの説明

`hand_amovement.py` のプログラムについて説明します。

## 1. このプログラムは何をするの？
このプログラムは **PCについているカメラを使って手の動きを検出し、動いている方向を表示** するものです。Pythonの **OpenCV** と **MediaPipe** を使って、手の位置を認識し、どの方向に動いているかを判断します。

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
python hand_amovement.py
```

PCのカメラ映像が表示されるようになります。
手をカメラに映すとカメラ映像に手の動きが表示されます。

**'q'キーを押すとプログラムが終了** します。

## 5. まとめ
- **Webカメラを使って手の動きを検出**
- **手の中心を求め、前の位置と比較して動く方向を判定**
- **結果を画面にリアルタイム表示**
- **'q'キーを押すとプログラム終了**

---

# 手のジェスチャーを認識するプログラムの説明

`hand_gesture.py` のプログラムについて説明します。

## 1. このプログラムでできること
このプログラムは、**カメラを使って手の形（グー・チョキ・パー）を認識** します。
- **グー（rock）** : すべての指を閉じた状態
- **チョキ（scissors）** : 人差し指と中指だけ開いた状態
- **パー（paper）** : すべての指を開いた状態

カメラに手を映すと、リアルタイムでジェスチャーを判定し、画面に結果を表示します。

## 2. 必要なライブラリ
プログラムを実行する前に、以下のライブラリをインストールしてください。
```bash
pip install opencv-python mediapipe numpy
```

## 3. プログラムの動作の仕組み

### (1) ライブラリのインポート
```python
import cv2
import mediapipe as mp
import numpy as np
```
このプログラムでは、以下のライブラリを使用します。
- **OpenCV（cv2）** : 画像処理やカメラの映像取得
- **MediaPipe（mp）** : 手のランドマーク（特徴点）を検出
- **NumPy（np）** : 配列計算に利用

### (2) MediaPipeの初期化
```python
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
```
MediaPipeの手の検出を設定し、**手のランドマークを描画** するための準備を行います。

```python
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
```
- **max_num_hands=1** → 検出する手は最大1つ
- **min_detection_confidence=0.7** → 検出の信頼度が70%以上なら認識
- **min_tracking_confidence=0.7** → 追跡の信頼度が70%以上なら追跡

### (3) 指の形状を判定する関数
```python
def recognize_gesture(landmarks):
```
指の関節の **y座標** を比較して、指が開いているか閉じているかを判定します。

- **親指の判定**
  ```python
  fingers.append(landmarks[4].x < landmarks[3].x)
  ```
  → **左手と右手でx座標の判定が逆** になることに注意。

- **人差し指～小指の判定**
  ```python
  fingers.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)
  ```
  → **指先のy座標が第二関節より上にあるかどうか** で判定。

- **ジェスチャーを識別**
  ```python
  if all(fingers[1:]):
      return "paper"
  elif not any(fingers):
      return "rock"
  elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
      return "scissors"
  return "nothing"
  ```
  - **すべての指が開いていれば「パー」**
  - **すべての指が閉じていれば「グー」**
  - **人差し指と中指だけ開いていれば「チョキ」**

### (4) カメラの設定
```python
cap = cv2.VideoCapture(0)
```
- **カメラを起動** して映像を取得します。
- 画像サイズを **半分に縮小** して処理の負荷を減らします。
  ```python
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  ```

### (5) メインループ（カメラ映像の処理）
```python
while cap.isOpened():
```
**リアルタイムでカメラ映像を取得し、手のジェスチャーを認識** します。

1. **画像をRGBに変換**
   ```python
   image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   ```
   OpenCVはBGR形式を使うため、MediaPipeで処理できるようにRGBに変換します。

2. **手のランドマークを検出**
   ```python
   results = hands.process(image)
   ```
   `hands.process(image)` を使って手のランドマークを取得します。

3. **手のランドマークを描画**
   ```python
   if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
           mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
   ```
   → **手の関節を画面に表示** します。

4. **ジェスチャーを判定し、画面に表示**
   ```python
   gesture = recognize_gesture(hand_landmarks.landmark)
   cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
   ```
   → **認識したジェスチャーを映像にテキストとして表示** します。

### (6) プログラムの終了
```python
if cv2.waitKey(10) & 0xFF == ord('q'):
    break
```
- **'q'キーを押すとプログラムが終了** します。

## 4. 実行方法
1. ターミナルまたはコマンドプロンプトで、以下のコマンドを実行します。
   ```bash
   python hand_gesture.py
   ```
2. **カメラの前で手を動かす** と、リアルタイムで「グー」、「チョキ」、「パー」を判定します。
3. **終了するときは 'q'キー を押します。**

## 5. まとめ
- **手のランドマークを検出** して、指が開いているか閉じているかを判定
- **「グー」、「チョキ」、「パー」をリアルタイムで認識**
- **画面に結果を表示** する
