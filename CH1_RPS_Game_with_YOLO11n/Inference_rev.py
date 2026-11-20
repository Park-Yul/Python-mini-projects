# -*- coding: utf-8 -*-
# ============================================================
#  Rock-Paper-Scissors 2-Player Game  (LiteRT / TFLite + OpenCV)
#   - 모델        : RPS_Project_int8.tflite  (입력 1x320x320x3, 출력 1x7x2100)
#   - 카메라      : 640x480
#   - 화면        : 세로 중앙선 기준 좌 = Player 1, 우 = Player 2
#   - 표시        : 플레이어별 색상 Bounding Box + "P1/P2 - 클래스" 라벨
#   - 상단 중앙   : 실시간 승패 결과
#   - 종료        : q 키
# ============================================================
import os
import time
from collections import Counter, deque

import cv2
import numpy as np

# ---------- LiteRT 인터프리터 로딩 (설치된 것 중 하나를 자동 선택) ----------
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite  # type: ignore

# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
MODEL_NAME = "RPS_Project_int8.tflite"
IMG_SIZE = 320            # 모델 입력 크기
CAM_W, CAM_H = 640, 480   # 카메라 해상도
MIRROR = False            # True면 좌우 반전(거울 모드)

CONF_TH = 0.5             # 신뢰도 임계값
IOU_TH = 0.45             # NMS IoU 임계값
VOTE_LEN = 5              # 결과 안정화용 최근 프레임 수 (다수결)
VOTE_MIN = 3              # 다수결 최소 득표 수

ansToText = {0: "scissors", 1: "rock", 2: "paper"}
SCISSORS, ROCK, PAPER = 0, 1, 2
PLAYER_COLOR = {1: (255, 120, 0), 2: (0, 120, 255)}  # BGR (P1: 파랑계, P2: 주황계)

# ------------------------------------------------------------
# 모델 로딩
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
modelPath = os.path.join(script_dir, MODEL_NAME)
if not os.path.exists(modelPath):          # 스크립트 폴더에 없으면 현재 폴더에서 찾기
    modelPath = MODEL_NAME
print("model path:", modelPath)

interpreter = tflite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]
output_dtype = output_details[0]["dtype"]
input_scale, input_zero = input_details[0]["quantization"]
output_scale, output_zero = output_details[0]["quantization"]
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]
print("model input shape:", (height, width))
print("input dtype:", input_dtype.__name__, "/ output dtype:", output_dtype.__name__)


# ------------------------------------------------------------
# 전처리: letterbox (비율 유지 + 패딩)  -> 640x480도 왜곡 없이 처리
# ------------------------------------------------------------
def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = nw - new_w, nh - new_h
    pad_x, pad_y = pad_w // 2, pad_h // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_y, pad_h - pad_y,      # 홀수 패딩도 정확히 맞춤
        pad_x, pad_w - pad_x,
        cv2.BORDER_CONSTANT, value=color,
    )
    return padded, r, pad_x, pad_y


# ------------------------------------------------------------
# 모델 추론 + 디코딩  (반환: [(x1,y1,x2,y2), cls_id, score] 리스트)
# ------------------------------------------------------------
def detect(frame):
    fh, fw = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_lb, r, pad_x, pad_y = letterbox(img_rgb)
    img = img_lb.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)                       # (1,320,320,3)

    # 입력 dtype에 맞게 변환 (float32 / int8 / uint8 모두 대응)
    if input_dtype == np.int8 or input_dtype == np.uint8:
        info = np.iinfo(input_dtype)
        img = np.clip(np.round(img / input_scale + input_zero), info.min, info.max).astype(input_dtype)
    else:
        img = img.astype(input_dtype)

    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_index)[0]

    # 출력 dtype이 정수형이면 역양자화
    if output_dtype == np.int8 or output_dtype == np.uint8:
        raw = (raw.astype(np.float32) - output_zero) * output_scale
    raw = raw.astype(np.float32)

    # (7,2100) -> (2100,7)
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T

    boxes_xywh = raw[:, :4].copy()
    cls_scores = raw[:, 4:]

    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(len(cls_ids)), cls_ids]

    mask = scores >= CONF_TH
    if not np.any(mask):
        return []
    boxes_xywh, cls_ids, scores = boxes_xywh[mask], cls_ids[mask], scores[mask]

    # 좌표가 0~1 정규화 값이면 픽셀로 변환
    if boxes_xywh.max() <= 2.0:
        boxes_xywh *= IMG_SIZE

    cx, cy, w, h = boxes_xywh.T
    x1 = np.clip((cx - w / 2 - pad_x) / r, 0, fw - 1)
    y1 = np.clip((cy - h / 2 - pad_y) / r, 0, fh - 1)
    x2 = np.clip((cx + w / 2 - pad_x) / r, 0, fw - 1)
    y2 = np.clip((cy + h / 2 - pad_y) / r, 0, fh - 1)

    nms_boxes = [[int(a), int(b), int(c - a), int(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    keep = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), CONF_TH, IOU_TH)
    keep = np.array(keep).flatten().astype(int) if len(keep) else []

    dets = []
    for i in keep:
        bx1, by1, bx2, by2 = int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])
        if bx2 - bx1 < 5 or by2 - by1 < 5:
            continue
        dets.append(((bx1, by1, bx2, by2), int(cls_ids[i]), float(scores[i])))
    return dets


# ------------------------------------------------------------
# Bounding Box 그리기 (플레이어별 색상 + 라벨)
# ------------------------------------------------------------
def draw_box(frame, box, color, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 라벨 (박스 위쪽에 배경과 함께)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(y1, th + 8)
    tx = min(x1, frame.shape[1] - tw - 6)
    cv2.rectangle(frame, (tx, ty - th - 8), (tx + tw + 6, ty + bl - 4), color, -1)
    cv2.putText(frame, label, (tx + 3, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ------------------------------------------------------------
# 게임 로직
# ------------------------------------------------------------
history = {1: deque(maxlen=VOTE_LEN), 2: deque(maxlen=VOTE_LEN)}


def stable_gesture(player):
    votes = [g for g in history[player] if g is not None]
    if not votes:
        return None
    gid, cnt = Counter(votes).most_common(1)[0]
    return gid if cnt >= VOTE_MIN else None


def judge(g1, g2):
    """반환: 'P1 WIN' / 'P2 WIN' / 'DRAW' / None"""
    if g1 is None or g2 is None:
        return None
    if g1 == g2:
        return "DRAW"
    # scissors(0) < rock(1) < paper(2) < scissors : (a-b)%3==1 이면 a 승
    return "PLAYER 1 WIN!" if (g1 - g2) % 3 == 1 else "PLAYER 2 WIN!"


def put_center_text(img, text, cx, y, scale, color, thick=2, bg=(0, 0, 0)):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(cx - tw / 2)
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + bl + 4), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def processImage(frame):
    fh, fw = frame.shape[:2]
    mid = fw // 2

    dets = detect(frame)

    # 플레이어별(박스 중심 x 기준) 가장 신뢰도 높은 검출 1개 선택
    best = {1: None, 2: None}
    for box, cid, sc in dets:
        cx = (box[0] + box[2]) / 2
        p = 1 if cx < mid else 2
        if best[p] is None or sc > best[p][2]:
            best[p] = (box, cid, sc)

    # Bounding Box 표시
    for p in (1, 2):
        history[p].append(best[p][1] if best[p] else None)
        if best[p]:
            box, cid, sc = best[p]
            draw_box(frame, box, PLAYER_COLOR[p], f"P{p} - {ansToText[cid]} {sc:.2f}")

    # 중앙 세로선
    cv2.line(frame, (mid, 0), (mid, fh), (255, 255, 255), 2)

    # 상단: 플레이어 이름 + 현재 제스처
    g = {1: stable_gesture(1), 2: stable_gesture(2)}
    cv2.putText(frame, "PLAYER 1", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PLAYER_COLOR[1], 2)
    cv2.putText(frame, "PLAYER 2", (mid + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PLAYER_COLOR[2], 2)
    for p, x0 in ((1, 10), (2, mid + 10)):
        txt = ansToText[g[p]] if g[p] is not None else "-"
        cv2.putText(frame, txt, (x0, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 상단 중앙: 실시간 승패
    result = judge(g[1], g[2])
    if result is None:
        put_center_text(frame, "Waiting...", mid, 90, 0.8, (200, 200, 200))
    elif result == "DRAW":
        put_center_text(frame, "DRAW", mid, 90, 1.0, (0, 255, 255))
    else:
        col = PLAYER_COLOR[1] if "1" in result else PLAYER_COLOR[2]
        put_center_text(frame, result, mid, 90, 1.0, col)


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cam", CAM_W + 40, CAM_H + 60)

    startTime = time.time()
    while cap.isOpened():
        ret, frame = cap.read()          # (480, 640, 3)
        if not ret:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)

        processImage(frame)

        curTime = time.time()
        fps = 1 / max(curTime - startTime, 1e-6)
        startTime = curTime
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        cv2.imshow("cam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
