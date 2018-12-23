# coding : utf-8

# from picamera.array import PiRGBArray
# from picamera import PiCamera
import numpy as np
import time
import cv2
import os
# import pygame
import screeninfo
import math
import copy
from multibars import MagnetBar, MultiBars

BWrev = 1

# set up PI camera
# camera = PiCamera()
# camera.resolution = (320, 180)
# camera.framerate = 40
# rawCapture = PiRGBArray(camera, size=(320, 180))

px = 1280
py = 720
cube = 160

# 绘制需要投影的棋盘格图像
ChessBoardImg = np.zeros((py, px, 3), np.uint8)
BalckBoard = np.zeros((py, px, 3), np.uint8)
blackcolor = np.array([255, 255, 255])
for j in range(ChessBoardImg.shape[1]):
    for i in range(ChessBoardImg.shape[0]):
        if (i // cube + j // cube) % 2:
            ChessBoardImg[i, j, :] = blackcolor
# cv2.imwrite('ChessBoard1.jpg', ChessBoardImg)
if BWrev:
    ChessBoardImg = 255 - ChessBoardImg

# Calibrate Camera seting
# 棋盘格角点个数 M * N
M = px // cube - 1
N = py // cube
# 棋盘格第一个点的投影坐标与各向的间隔
ChessBoard_X0 = cube
ChessBoard_Y0 = cube
ChessBoard_deltaX = cube
ChessBoard_deltaY = cube

# 所有标定板角点的实际尺寸位置坐标，三维，z=0
objp = np.zeros((M * N, 3), np.float32)
ChessBoardPane = np.mgrid[0:M, 0:N].T.reshape(-1, 2)
ChessBoardPane = ChessBoardPane.astype(np.float64)
ChessBoardPane[:, 0] = ChessBoardPane[:, 0] * ChessBoard_deltaX + ChessBoard_X0
ChessBoardPane[:, 1] = ChessBoardPane[:, 1] * ChessBoard_deltaY + ChessBoard_Y0
objp[:, :2] = ChessBoardPane


def MySwap(obj_points, img_points):
    new_obj_points = copy.deepcopy(img_points)
    new_img_points = copy.deepcopy(obj_points)
    for i in range(np.array(obj_points).shape[1]):
        new_img_points[0][i, 0] = img_points[0][i, 0, 0]
        new_img_points[0][i, 1] = img_points[0][i, 0, 1]
        new_obj_points[0][i, 0, 0] = obj_points[0][i, 0]
        new_obj_points[0][i, 0, 1] = obj_points[0][i, 1]
    return new_obj_points, new_img_points


# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
screen = screeninfo.get_monitors()[1]
cap = cv2.VideoCapture(1)


def CalibrateCamera():
    window_name = "ChessBoard"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while 1:
        cv2.imshow(window_name, ChessBoardImg)
        cv2.waitKey(100)
        cap.read()
        cap.read()

        ret, image = cap.read()
        # cv2.imwrite("camera.jpg", image)
        # image = cv2.imread("camera.jpg")
        # image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if BWrev:
            gray = 255 - gray

        # Calibrate camera
        ret, corners = cv2.findChessboardCorners(gray, (M, N), None)

        # find enough key point to caliberate the camera
        if ret:
            obj_points = []  # 存储3D点，所有格点实际尺寸标注位置
            img_points = []  # 存储2D点，照片中的像素点位置
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            img_points.append(corners2)
            cv2.drawChessboardCorners(image, (M, N), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            cv2.imwrite('camera.jpg', image)
        else:
            print("Cannot detected the chessboard!")
            cv2.imwrite('camera.jpg', gray)
            # break
            continue

        # print('img_points = ')
        # print(img_points)
        # print('obj_points = ')
        # print(obj_points)
        # print('*******************')
        img_points, obj_points, = MySwap(obj_points, img_points)
        # print('img_points = ')
        # print(img_points)
        # print('obj_points = ')
        # print(obj_points)
        # 标定main function
        Size = gray.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, Size, None, None)
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, Size, None, None)
        print('Succeed ! ')
        np.savez('Init.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

        cap.release()
        cv2.destroyAllWindows()
        return rvecs, tvecs, mtx, dist


def cv_distance(P, Q):
    return int(math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]), 2)))


def calc_magnet(center, img, rvecs, tvecs, mtx, dist):
    LC = np.array([cv_distance(center[0], center[1]), cv_distance(center[1], center[2]),
                   cv_distance(center[0], center[2])])
    # print(LC)
    LC_max = np.argmax(LC, axis=0)
    LC_min = np.argmin(LC, axis=0)
    LC = [[0, 1], [1, 2], [0, 2]]
    poleN = set(LC[LC_max]).intersection(set(LC[LC_min]))
    poleM = set(LC[LC_min]).difference(poleN)
    poleS = set(LC[LC_max]).difference(poleN)
    poleN = list(poleN)[0]
    poleM = list(poleM)[0]
    poleS = list(poleS)[0]

    cam_point = np.array([[center[poleN][0], center[poleN][1], 0],
                          [center[poleS][0], center[poleS][1], 0]],
                         dtype=np.float32)

    img_points, _ = cv2.projectPoints(cam_point, rvecs[0], tvecs[0], mtx, dist)
    img_points = img_points.reshape(-1, 2)
    # print(img_points)
    # Nx = img_points[0, 0]
    # Ny = img_points[0, 1]
    # Sx = img_points[1, 0]
    # Sy = img_points[1, 1]

    # cv2.circle(img, center[poleN], 2, (255, 0, 0), 5, cv2.LINE_AA)
    # cv2.circle(img, center[poleM], 2, (0, 255, 0), 5, cv2.LINE_AA)
    # cv2.circle(img, center[poleS], 2, (0, 0, 255), 5, cv2.LINE_AA)

    N = (img_points[0, 0], img_points[0, 1])
    S = (img_points[1, 0], img_points[1, 1])

    length = math.sqrt(pow((N[0] - S[0]), 2) + pow((N[1] - S[1]), 2))
    VSN = ((N[0] - S[0]), (N[1] - S[1]))
    V90 = (VSN[1], -VSN[0])
    N = np.array(N)
    S = np.array(S)
    VSN = np.array(VSN)
    V90 = np.array(V90)
    node = np.zeros(8, dtype=float).reshape(4, 2)
    node[0, :] = N + VSN / 5 * 3 - V90 / 5
    node[1, :] = S - VSN / 5 * 3 - V90 / 5
    node[2, :] = S - VSN / 5 * 3 + V90 / 5
    node[3, :] = N + VSN / 5 * 3 + V90 / 5

    center = ((N[0] + S[0]) / 2, (N[1] + S[1]) / 2)

    theta = np.arccos(- VSN[0] / length)
    if VSN[1] < 0:
        theta = 3.1415926 * 2 - theta

    length = length / 5 * 11
    width = length / 5 * 2

    return node.astype(int), theta, center, length, width


def JudgeGreen(img, center):
    step = 1
    point = np.zeros((9, 2), dtype=int)
    point[0] = [center[0], center[1]]
    point[1] = [center[0] + step, center[1]]
    point[2] = [center[0] - step, center[1]]
    point[3] = [center[0], center[1] + step]
    point[4] = [center[0], center[1] - step]
    point[5] = [center[0] + step, center[1] + step]
    point[6] = [center[0] - step, center[1] - step]
    point[7] = [center[0] + step, center[1] - step]
    point[8] = [center[0] - step, center[1] + step]
    cnt = 0
    orimg = img
    for p in point:
        b = img[p[1], p[0]][0]
        g = img[p[1], p[0]][1]
        r = img[p[1], p[0]][2]
        if g > r and g > b and g - b > 20 and g - r > 40 and g > 100:
            cnt = cnt + 1
    if cnt >= 6:
        return True
    else:
        return False


def Projected(img=BalckBoard):
    window_name = "BalckBoard"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    cv2.waitKey(10)


def detect_magnet(mtx, dist, rvecs, tvecs):
    BalckBoard = np.zeros((py, px, 3), np.uint8)
    center_save = (0, 0)
    green_center_save = (0, 0)
    black_center_save = (0, 0)
    # begin_key = 1
    cap = cv2.VideoCapture(1)
    while 1:
        print('Begin ***')
        # get a frame
        ret, img = cap.read()
        # show a frame
        # if begin_key:
        #     # cv2.imshow("capture", img)
        #     # cv2.waitKey(1)
        #     begin_key = 0
        # cv2.imshow("capture", img)
        # cv2.waitKey(10)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gb = cv2.GaussianBlur(img_gray, (5, 5), 0)
        edges = cv2.Canny(img_gray, 100, 200)

        img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hierarchy = hierarchy[0]
        found = []
        for i in range(len(contours)):
            k = i
            c = 0
            while hierarchy[k][2] != -1:
                k = hierarchy[k][2]
                c = c + 1
            if c >= 5:
                found.append(i)

        if len(found) < 3:
            continue

        boxes = []
        center = []
        rectheight = []
        sidelen = 0
        for i in found:
            rect = cv2.minAreaRect(contours[i])
            center.append((int(rect[0][0]), int(rect[0][1])))
            rectheight.append(rect[1][1])
            sidelen = sidelen + rect[1][0]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = map(tuple, box)
            boxes.append(box)

        sidelen = sidelen / len(found)
        # numbars = 0

        if len(found) < 3:
            # print('No Enough rect detected! ')
            continue
        else:
            savemark = np.ones(len(found))
            for i in range(len(found)):
                if savemark[i] == 1:
                    tmpstore = []  # store the repeat rectangle
                    for j in np.arange(i + 1, len(found)):
                        if cv_distance(center[i], center[j]) < sidelen / 4:
                            tmpstore.append(j)
                    if len(tmpstore) == 0:
                        pass
                    else:
                        tag = i
                        best = rectheight[tag]
                        for k in tmpstore:
                            if best > rectheight[k]:
                                savemark[k] = 0
                            else:
                                tag = k
                                best = rectheight[tag]
                                savemark[i] = 0
            boxeses = []
            centers = []
            for i in range(len(found)):
                if savemark[i] == 1:
                    boxeses.append(boxes[i])
                    centers.append(center[i])
            boxes = boxeses
            center = centers

        if len(center) == 3:
            numbars = 1
        elif len(center) == 6:
            numbars = 2
        else:
            continue
            # print('No Enough rect detected! ')
        # print(center)
        # numbars = 1
        # print(numbars)
        if numbars == 2:
            green_center = []
            black_center = []
            green_mark = np.zeros(6, dtype=int)
            for i in range(len(center)):
                if JudgeGreen(img, center[i]):
                    green_mark[i] = 1
            if np.sum(green_mark) == 3:
                for i in range(len(center)):
                    if green_mark[i] == 1:
                        green_center.append(center[i])
                    else:
                        black_center.append(center[i])
            else:
                continue

            proj_img = np.zeros((py, px, 3), np.uint8)
            print('Calcating **** ')
            node1, theta1, center1, length1, width1 = calc_magnet(green_center, proj_img, rvecs, tvecs, mtx, dist)
            node2, theta2, center2, length2, width2 = calc_magnet(black_center, proj_img, rvecs, tvecs, mtx, dist)
            print(cv_distance(green_center_save, center1) + cv_distance(black_center_save, center2))
            if cv_distance(green_center_save, center1) + cv_distance(black_center_save, center2) < 10:
                print('save time 2')

                continue

            green_center_save = center1
            black_center_save = center2
            print('Projecting **** ')
            bar1 = MagnetBar(length1, width1, center1, theta1, proj_img)
            bar2 = MagnetBar(length2, width2, center2, theta2, proj_img)
            bars = []
            bars.append(bar1)
            bars.append(bar2)
            multi_bar = MultiBars(bars, proj_img)
            multi_bar.draw_magnets()
            print('Projecting Finish **** ')
        else:
            proj_img = np.zeros((py, px, 3), np.uint8)

            node, theta, center, length, width = calc_magnet(center, proj_img, rvecs, tvecs, mtx, dist)
            print(cv_distance(center_save, center))
            if cv_distance(center_save, center) < 5:
                print('save time 1')

                continue

            center_save = center
            bar1 = MagnetBar(length, width, center, theta, proj_img)
            bars = []
            bars.append(bar1)
            multi_bar = MultiBars(bars, proj_img)
            multi_bar.draw_magnets()
        Projected(proj_img)
        # cv2.imshow("capture", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # node, theta, center, length, width = calc_magnet((Nx, Ny), (Sx, Sy))
    #
    #
    # bar1 = MagnetBar(length, width, center, theta, img)
    # bars = []
    # bars.append(bar1)
    # multi_bar = MultiBars(bars, img)
    # multi_bar.draw_magnets()
    # # cv2.line(img, (node[0, 0], node[0, 1]), (node[1, 0], node[1, 1]), (0, 0, 255), 5, cv2.LINE_AA)
    # # cv2.line(img, (node[2, 0], node[2, 1]), (node[1, 0], node[1, 1]), (0, 0, 255), 5, cv2.LINE_AA)
    # # cv2.line(img, (node[2, 0], node[2, 1]), (node[3, 0], node[3, 1]), (0, 0, 255), 5, cv2.LINE_AA)
    # # cv2.line(img, (node[0, 0], node[0, 1]), (node[3, 0], node[3, 1]), (0, 0, 255), 5, cv2.LINE_AA)
    # Projected(img)
    # BalckBoard = np.zeros((py, px, 3), np.uint8)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    mode = 1
    if mode == 0:
        rvecs, tvecs, mtx, dist = CalibrateCamera()
        print(rvecs[0], tvecs[0], mtx, dist)
    else:
        with np.load('Init.npz') as X:
            mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        # print(mtx, dist, rvecs, tvecs)
        Projected()
        detect_magnet(mtx, dist, rvecs, tvecs)
