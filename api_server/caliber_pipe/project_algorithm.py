# -*- coding: utf-8 -*-
# center점으로부터 상하선분에 접하는 선분을 찾아서 거리측정
import numpy as np
import cv2 as cv
import math
import imutils
import logging


logging.basicConfig(filename='ImageProcessing.log',
                    level=logging.DEBUG, format='%(asctime)s:%(message)s')


FOCAL_LENGTH = 1410
WIDTH = 8

top_line_x1, top_line_y1, top_line_x2, top_line_y2 = 0, 0, 0, 0
bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = 0, 0, 0, 0
left_top_line_x1, left_top_line_y1, left_top_line_x2, left_top_line_y2 = 0, 0, 0, 0
curve_start_line_x1, curve_start_line_y1, curve_start_line_x2, curve_start_line_y2 = 0, 0, 0, 0

kernel = np.ones((3, 1), np.uint8)
kernel1 = np.ones((1, 3), np.uint8)


def distance_calculate(pixel):
    d = WIDTH * FOCAL_LENGTH / pixel
    return d


def line_point(line_list):

    for i in range(len(line_list[1:])):
        a1 = line_list[i][3] - line_list[i][1]
        b1 = line_list[i][0] - line_list[i][2]
        c1 = a1*line_list[i][0] + b1*line_list[i][1]
        a2 = line_list[i+1][3] - line_list[i+1][1]
        b2 = line_list[i+1][0] - line_list[i+1][2]
        c2 = a2*line_list[i+1][0] + b2*line_list[i+1][1]
        deteminate = a1*b2 - a2*b1
        try:
            t_X = int((b2*c1 - b1*c2)/deteminate)
            t_Y = int((a1*c2 - a2*c1)/deteminate)
        except ZeroDivisionError:
            t_X = int((b2*c1 - b1*c2)/1)
            t_Y = int((a1*c2 - a2*c1)/1)

    return t_X, t_Y


def threshold_func(img):

    frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    threshold = cv.inRange(frame_HSV, (60, 0, 0), (179, 255, 255))
    threshold2 = cv.inRange(img, (0, 0, 146), (179, 255, 255))
    threshold2 = cv.bitwise_not(threshold2)
    result1 = cv.erode(threshold, kernel, iterations=3)
    result1 = cv.dilate(result1, kernel, iterations=5)
    threshold_result = cv.bitwise_and(threshold, threshold2)
    threshold_result2 = cv.bitwise_and(result1, threshold_result)
    threshold_result2 = cv.dilate(threshold_result2, kernel1, iterations=5)
    return threshold_result2


def detect_center(roi_thres):
    cnts = cv.findContours(
        roi_thres.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        c = max(cnts, key=cv.contourArea)
        M = cv.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            cX = int(M["m10"] / 1)
            cY = int(M["m01"] / 1)
    return cX, cY


def main(PATH):
    img = cv.imread(PATH)
    img = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5,
                    interpolation=cv.INTER_LINEAR)
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    if img is not None:
        depth, pipe_depth, degree, pipe_type = object_detect(img)
        return depth, pipe_depth, degree, pipe_type
    else:
        logging.warning("There is No Image")
        return 0, 0, 0, 0


def object_detect(img):
    pipe_type = 0

    global left_top_line_x1, left_top_line_y1, left_top_line_x2, left_top_line_ygray2
    global top_line_x1, top_line_y1, top_line_x2, top_line_y2

    global bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2
    global curve_start_line_x1, curve_start_line_y1, curve_start_line_x2, curve_start_line_y2
    top_line_x1, top_line_y1, top_line_x2, top_line_y2 = 0, 0, 0, 0
    bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = 0, 0, 0, 0
    left_top_line_x1, left_top_line_y1, left_top_line_x2, left_top_line_y2 = 0, 0, 0, 0
    curve_start_line_x1, curve_start_line_y1, curve_start_line_x2, curve_start_line_y2 = 0, 0, 0, 0

    threshold_result2 = threshold_func(img)
    cnts = cv.findContours(
        threshold_result2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    # DEFINE: x,y좌표가 0일 경우에 좌표를 더 늘리거나 추가하는 작업에서 오류가 발생하여서 분개함
    if x is 0 and y is 0:
        roi_image = img[1:1+h, 1:1+w]
    else:
        roi_image = img[y-10:y+h+10, x:x+w]

    roi_thres = threshold_func(roi_image)
    cX, cY = detect_center(roi_thres)
    cv.circle(roi_image, (cX, cY), 7, (255, 0, 255), -1)
    edges = cv.Canny(roi_thres, 100, 200, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100,
                           minLineLength=0, maxLineGap=500)
    line_list = list()
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 < cY and y2 < cY and x1 < cX:
                top_line_x1, top_line_y1, top_line_x2, top_line_y2 = x1, y1, x2, y2
            elif y1 > cY and y2 > cY and x1 < cX:
                bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = x1, y1, x2, y2

            elif (y2-y1)/(x2-x1) < 0:
                curve_start_line_x1, curve_start_line_y1, curve_start_line_x2, curve_start_line_y2 = x1, y1, x2, y2
                cv.line(roi_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        line_list.append([bottom_line_x1, bottom_line_y1,
                          bottom_line_x2, bottom_line_y2])

        line_list.append([curve_start_line_x1, curve_start_line_y1,
                          curve_start_line_x2, curve_start_line_y2])

        if curve_start_line_x1 is not 0:
            t_X, t_Y = line_point(line_list)
            cv.circle(roi_image, (t_X, t_Y), 7, (0, 255, 255), -1)
            top_degree_height = math.atan2(
                (bottom_line_x1 - t_X), (bottom_line_y1 - t_Y))
            top_degree_bottom = math.atan2(
                (t_X-curve_start_line_x1), (t_Y-curve_start_line_y1))
            degree = (top_degree_height-top_degree_bottom)*180/math.pi
            print(degree)
            if degree < 0:
                print("asdfasdfsdf")
                degree = abs(degree)
                if degree > 180:
                    degree = degree-180
                if degree < 170:
                    pipe_type = 1
            if degree > 175:
                degree = 0
            # if degree > 175:
            #     pipe_type = 1
            #     degree = 0

        else:
            degree = 0
            pipe_type = 0
            print("degree", degree)

        pixel = bottom_line_y1 - top_line_y1
        distance = distance_calculate(pixel)
        print(distance, pixel, degree, pipe_type)
        logging.debug({
            'DepthToPipe': distance,
            'PixelDistance': pixel,
            'Type': pipe_type,
            'Degree': degree
        })
        if pipe_type is 0:
            return distance, pixel, degree, pipe_type
        else:
            return 0, 0, degree, pipe_type
    except:
        logging.debug("There's no Line")
        return 0, 0, 0, 0
