# -*-coding: utf-8 -*-
import numpy as np
import cv2

lower = np.array([0, 48, 80], dtype='uint8')
upper = np.array([20, 255, 255], dtype='uint8')

camera = cv2.VideoCapture('video.mp4')
# size = (camera.col,camera.row)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.mp4', fourcc, 25.0, (int(camera.get(3)), int(camera.get(4))))

# 전역 변수
(grabbed, frame) = camera.read()
roi = frame[:, :]
switch = True

ix1, ix2 = 0, camera.get(3)
iy1, iy2 = 0, camera.get(4)


# Mouse Callback함수
def set_roi(event, x, y, flags, param):
    global ix1, iy1, ix2, iy2, drawing

    if event == cv2.EVENT_LBUTTONDOWN: # 마우스를 누른 상태
        drawing = True
        ix1, iy1 = x, y

    # elif event == cv2.EVENT_MOUSEMOVE: # 마우스 이동
    # if drawing == True:            # 마우스를 누른 상태 일경우
    # cv2.rectangle(img,(ix,iy),(x,y),(255,0,0),-1)

    elif event == cv2.EVENT_LBUTTONUP: # 마우스를 뗀 상태
        drawing = False             # 마우스를 떼면 상태 변경
        ix2, iy2 = x, y
        print(ix1, ' ', ix2, ' ', iy1, ' ', iy2)


cv2.imshow('images', frame)
cv2.setMouseCallback('images', set_roi)

# 영상 시작
print('영상 시작\n')
while True:
    if switch:
        (grabbed, frame) = camera.read()

        if grabbed:
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skinMask = cv2.inRange(converted, lower, upper)

            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            # skinMask = cv2.erode(skinMask, kernel, iterations=1)
            # skinMask = cv2.dilate(skinMask, kernel, iterations=1)

            # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
            skin = cv2.bitwise_and(frame, frame, mask=skinMask)

            # 피부 색깔 변환
            skin[:, :, 2] = 0

            # 마스킹 및 영상 융합
            skin_gray = skin.copy()
            skin_gray = cv2.cvtColor(skin_gray, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(skin_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            roi = frame[int(iy1):int(iy2), int(ix1):int(ix2), :]
            frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv[int(iy1):int(iy2), int(ix1):int(ix2)])
            roi = skin[int(iy1):int(iy2), int(ix1):int(ix2), :]
            skin_fg = cv2.bitwise_and(roi, roi, mask=mask[int(iy1):int(iy2), int(ix1):int(ix2)])

            dst = cv2.add(frame_bg, skin_fg)
            frame[int(iy1):int(iy2), int(ix1):int(ix2), :] = dst

            # 출력
            cv2.imshow('images', frame)

            # 저장
            out.write(frame)

    # 키 설정
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        switch = not switch


camera.release()
out.release()
cv2.destroyWindow('images')