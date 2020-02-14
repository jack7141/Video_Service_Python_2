# -*- coding: utf-8 -*-
# center점으로부터 상하선분에 접하는 선분을 찾아서 거리측정
import numpy as np
import cv2 as cv
import math
import imutils
import logging
logging.basicConfig(filename='ImageProcessing.log',
                    level=logging.DEBUG, format='%(asctime)s:%(message)s')

FOCAL_LENGTH = 1130
WIDTH = 10

left_top_line_x1, left_top_line_y1, left_top_line_x2, left_top_line_y2 = 0, 0, 0, 0
top_line_x1, top_line_y1, top_line_x2, top_line_y2 = 0, 0, 0, 0

bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = 0, 0, 0, 0
right_bottom_line_x1, right_bottom_line_y1, right_bottom_line_x2, right_bottom_line_y2 = 0, 0, 0, 0

kernel = np.ones((3, 1), np.uint8)
kernel1 = np.ones((1, 3), np.uint8)


def distance_calculate(pixel):
    d = WIDTH * FOCAL_LENGTH / pixel
    return d


def line_point():
    global left_top_line_x1, left_top_line_y1, left_top_line_x2, left_top_line_y2
    global top_line_x1, top_line_y1, top_line_x2, top_line_y2

    global bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2
    global right_bottom_line_x1, right_bottom_line_y1, right_bottom_line_x2, right_bottom_line_y2

    a1 = top_line_y2 - top_line_y1
    b1 = top_line_x1 - top_line_x2
    c1 = a1*top_line_x1 + b1*top_line_y1
    a2 = right_bottom_line_y2 - right_bottom_line_y1
    b2 = right_bottom_line_x1 - right_bottom_line_x2
    c2 = a2*right_bottom_line_x1 + b2*right_bottom_line_y1
    deteminate = a1*b2 - a2*b1

    try:
        t_X = int((b2*c1 - b1*c2)/deteminate)
        t_Y = int((a1*c2 - a2*c1)/deteminate)
    except ZeroDivisionError:
        t_X = int((b2*c1 - b1*c2)/1)
        t_Y = int((a1*c2 - a2*c1)/1)

    return t_X, t_Y


def main(PATH):
    img = cv.imread(PATH)
    if img is not None:
        depth, pipe_depth, degree, pipe_type = object_detect(img)
        return depth, pipe_depth, degree, pipe_type
    else:
        logging.warning("There is No Image")
        return 0, 0, 0, 0


def object_detect(img):
    pipe_type = 0

    global left_top_line_x1, left_top_line_y1, left_top_line_x2, left_top_line_y2
    global top_line_x1, top_line_y1, top_line_x2, top_line_y2

    global bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2
    global right_bottom_line_x1, right_bottom_line_y1, right_bottom_line_x2, right_bottom_line_y2

    blurred = cv.GaussianBlur(img, (5, 5), 0)
    frame_HSV = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    threshold = cv.inRange(frame_HSV, (70, 100, 0), (255, 255, 255))
    threshold2 = cv.inRange(img, (0, 0, 0), (255, 120, 100))
    result1 = cv.erode(threshold, kernel, iterations=3)
    result1 = cv.dilate(result1, kernel, iterations=10)
    threshold_result = cv.bitwise_and(threshold, threshold2)
    threshold_result2 = cv.bitwise_and(result1, threshold_result)
    threshold_result2 = cv.dilate(result1, kernel1, iterations=20)

    cnts = cv.findContours(
        threshold_result2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
        cv.circle(img, (cX, cY), 7, (0, 0, 255), -1)

    edges = cv.Canny(threshold_result2, 100, 200, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/170, 90,
                           minLineLength=0, maxLineGap=500)
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 상단
            if y1 > cY and x1 < cX:
                top_line_x1, top_line_y1, top_line_x2, top_line_y2 = x1, y1, x2, y2
            # 하단
            elif y1 < cY and x1 < cX and x1 < 400 and (y2-y1)/(x2-x1) < 0.1:
                bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = x1, y1, x2, y2

            elif (y2-y1)/(x2-x1) > 0.5:
                right_bottom_line_x1, right_bottom_line_y1, right_bottom_line_x2, right_bottom_line_y2 = x1, y1, x2, y2
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# LOGGING = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'handlers': {
#         'file': {
#             'level': 'DEBUG',
#             'class': 'logging.FileHandler',
#             'filename': 'debug.log',
#             },
#     },
#     'loggers': {
#         'django.request': {
#             'handlers': ['file'],
#             'level': 'ERROR',
#             'propagate': True,
#             },
#         },
#     }

        if right_bottom_line_x1 is not 0:
            t_X, t_Y = line_point()
            top_degree_height = math.atan2(
                (top_line_x1 - t_X), (top_line_y1 - t_Y))
            top_degree_bottom = math.atan2(
                (right_bottom_line_x1 - t_X), (right_bottom_line_y1 - t_Y))
            degree = (top_degree_height-top_degree_bottom)*180/math.pi
            pipe_type = 1
        else:
            degree = 0
            pipe_type = 0

        cv.line(img, (top_line_x1, top_line_y1),
                (top_line_x2, top_line_y2), (255, 0, 255), 2)
        cv.line(img, (bottom_line_x1, bottom_line_y1),
                (bottom_line_x2, bottom_line_y2), (255, 255, 255), 2)
        pixel = top_line_y1-bottom_line_y1
        distance = distance_calculate(pixel)
        logging.debug({
            'DepthToPipe': distance,
            'PixelDistance': pixel,
            'Type': pipe_type,
            'Degree': degree
        })
        return distance, pixel, degree, pipe_type
    except:
        logging.debug("There's no Line")
        return 0, 0, 0, 0
