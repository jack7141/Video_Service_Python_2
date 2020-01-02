# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import urllib
import numpy as np
from .detect_curve import detect_curve
kernel = np.ones((1, 3), np.uint8)
kernel1 = np.ones((3, 3), np.uint8)
kernelx = np.ones((3, 1), np.uint8)
kernel2 = np.ones((1, 20), np.uint8)
focal_lenth = 1250
pipe_diameter =0
edge_diameter =0
def pipe_calculate_depth(distance):
    return int(pipe_diameter * focal_lenth / distance)

def cutting_calculate_depth(distance):
    return int(edge_diameter * focal_lenth / distance)

def calculate_pipe(pipe_image):
    grab_gray = cv.cvtColor(pipe_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grab_gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]
    frame_HSV = cv.cvtColor(pipe_image, cv.COLOR_BGR2HSV)
    threshold = cv.inRange(
        frame_HSV, (0, 75, 50), (179, 255, 255))

    threshold = cv.erode(threshold, kernel1, iterations=10)
    threshold = cv.dilate(threshold, kernel1, iterations=10)
    threshold = cv.dilate(threshold, kernel2, iterations=10)
    pipe_edges = cv.Canny(threshold, 100, 150)
    lines = cv.HoughLines(pipe_edges, 1, np.pi/180, 150)

    top_line_x1, top_line_y1, top_line_x2, top_line_y2 = 0, 0, 0, 0
    bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = 0, 0, 0, 0
    min_y = 240
    max_y = 240
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if y1 < 250:
            if min_y > y1:
                min_y = y1
                top_line_x1, top_line_y1, top_line_x2, top_line_y2 = x1, y1, x2, y2
        else:
            if max_y < y1:
                max_y = y1
                bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = x1, y1, x2, y2

    cv.line(pipe_image, (top_line_x1, top_line_y1),
            (top_line_x2, top_line_y2), (255, 0, 255), 1)
    cv.line(pipe_image, (bottom_line_x1, bottom_line_y1),
            (bottom_line_x2, bottom_line_y2), (255, 0, 255), 1)
    depth = pipe_calculate_depth(bottom_line_y1-top_line_y1)
    return depth


def remove_noise(PATH,edge,pipe):
    global pipe_diameter
    global edge_diameter
    pipe_diameter = pipe 
    edge_diameter = edge
    
    image = cv.imread(PATH)

    curve = detect_curve(image)
    original = image.copy()

    grab_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grab_gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]
    soble_bitwise = cv.bitwise_not(thresh)
    result = cv.erode(soble_bitwise, kernel, iterations=5)
    result = cv.dilate(result, kernel1, iterations=30)
    contours, _ = cv.findContours(
        result.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img2 = image.copy()
    c = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    roi_image = img2[y:y+h, x:x+w]
    pipe_depth = calculate_pipe(roi_image)

    frame_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    threshold = cv.inRange(
        frame_HSV, (0, 0, 90), (179, 255, 255))
    thres_bitwise_and = cv.bitwise_and(thresh, threshold)
    thres_bitwise_and = cv.bitwise_not(thres_bitwise_and)
    thres_bitwise_and = cv.erode(thres_bitwise_and, kernel1, iterations=6)
    thres_bitwise_and = cv.dilate(thres_bitwise_and, kernel2, iterations=15)
    thres_bitwise_and = cv.dilate(thres_bitwise_and, kernel1, iterations=20)
    thres_bitwise_and = cv.dilate(thres_bitwise_and, kernelx, iterations=10)
    pipe_edges = cv.Canny(thres_bitwise_and, 100, 150)

    lines = cv.HoughLines(pipe_edges, 1, np.pi/180, 150)
    top_line_x1, top_line_y1, top_line_x2, top_line_y2 = 0, 0, 0, 0
    bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = 0, 0, 0, 0

    min_y = 280
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if y1 < 300:
            if min_y > y1:
                min_y = y1
                top_line_x1, top_line_y1, top_line_x2, top_line_y2 = x1, y1, x2, y2
        else:
            bottom_line_x1, bottom_line_y1, bottom_line_x2, bottom_line_y2 = x1, y1, x2, y2

    cv.line(original, (top_line_x1, top_line_y1),
            (top_line_x2, top_line_y2), (255, 0, 255), 1)
    cv.line(original, (bottom_line_x1, bottom_line_y1),
            (bottom_line_x2, bottom_line_y2), (255, 0, 255), 1)

    cutting_distance = abs(bottom_line_y1-top_line_y1)
    depth = cutting_calculate_depth(bottom_line_y1-top_line_y1)
    return depth, pipe_depth, curve
    cv.imshow("original", original)
    cv.waitKey()
    cv.destroyAllWindows()
