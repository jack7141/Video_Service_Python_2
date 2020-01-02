#-*- coding:utf-8-*-
import cv2 as cv
import numpy as np
import imutils
import math
kernel = np.ones((1, 3), np.uint8)
kernel1 = np.ones((3, 3), np.uint8)
kernelx = np.ones((3, 1), np.uint8)
kernel2 = np.ones((1, 20), np.uint8)
def detect_curve(img):
    # original_img = img
    kernel = np.ones((3,3), np.uint8)
    grab_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grab_gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]
    soble_bitwise = cv.bitwise_not(thresh)
    result = cv.erode(soble_bitwise, kernel, iterations=5)
    result = cv.dilate(result, kernel1, iterations=30)
    
    cnts = cv.findContours(result.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    angle_result = list()
    line_list = list()
    line_count = 0

    _9_x1,_9_y1,_9_x2,_9_y2 = 0, 0, 0, 0
    _8_x1,_8_y1,_8_x2,_8_y2 = 0, 0, 0, 0
    _7_x1,_7_y1,_7_x2,_7_y2 = 0, 0, 0, 0
    _6_x1,_6_y1,_6_x2,_6_y2 = 0, 0, 0, 0  
    _5_x1,_5_y1,_5_x2,_5_y2 = 0, 0, 0, 0  
    _4_x1,_4_y1,_4_x2,_4_y2 = 0, 0, 0, 0  
    _3_x1,_3_y1,_3_x2,_3_y2 = 0, 0, 0, 0  
    _2_x1,_2_y1,_2_x2,_2_y2 = 0, 0, 0, 0
    _1_x1,_1_y1,_1_x2,_1_y2 = 0, 0, 0, 0
    _0_x1,_0_y1,_0_x2,_0_y2 = 0, 0, 0, 0


    for c in cnts:
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    edges = cv.Canny(result,50,150)
    lines = cv.HoughLines(edges,1,np.pi/180,110)

    # FIXME: 선분 하나씩만 잡으려고 각 rho의 범위별로 분개
    # 별로 맘에 안듬 다른방법찾아야할듯
    for line in lines:

        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        if rho >= 900 and rho < 1000:
            _9_x1,_9_y1,_9_x2,_9_y2 = x1,y1,x2,y2
        if rho >= 800 and rho < 900:
            _8_x1,_8_y1,_8_x2,_8_y2 = x1,y1,x2,y2
        if rho >= 700 and rho < 800 :
            _7_x1,_7_y1,_7_x2,_7_y2 = x1,y1,x2,y2
        if rho >= 600 and rho < 700:
            _6_x1,_6_y1,_6_x2,_6_y2 = x1,y1,x2,y2
        if rho >= 500 and rho < 600:
            _5_x1,_5_y1,_5_x2,_5_y2 = x1,y1,x2,y2    
        if rho >= 500 and rho < 500:
            _4_x1,_4_y1,_4_x2,_4_y2 = x1,y1,x2,y2    
        if rho >= 300 and rho < 400:
            _3_x1,_3_y1,_3_x2,_3_y2 = x1,y1,x2,y2                    
        if rho >= 200 and rho < 300:
            _2_x1,_2_y1,_2_x2,_2_y2 = x1,y1,x2,y2
        if rho >= 100 and rho < 200:
            _1_x1,_1_y1,_1_x2,_1_y2 = x1,y1,x2,y2
        if rho > 0 and rho < 100:
            _0_x1,_0_y1,_0_x2,_0_y2 = x1,y1,x2,y2
            # if _1_x1 is None:
            #     _0_x1,_0_y1,_0_x2,_0_y2 = x1,y1,x2,y2
            #     print("sdfasdfasfsd:",_0_x1)
    
    cv.line(img,(_9_x1, _9_y1),(_9_x2, _9_y2),(0,0,255),2)
    cv.line(img,(_8_x1, _8_y1),(_8_x2, _8_y2),(0,0,255),2)
    cv.line(img,(_7_x1, _7_y1),(_7_x2, _7_y2),(0,0,255),2)
    cv.line(img,(_6_x1, _6_y1),(_6_x2, _6_y2),(0,0,255),2)
    cv.line(img,(_5_x1, _5_y1),(_5_x2, _5_y2),(0,0,255),2)
    cv.line(img,(_4_x1, _4_y1),(_4_x2, _4_y2),(0,0,255),2)
    cv.line(img,(_3_x1, _3_y1),(_3_x2, _3_y2),(0,0,255),2)
    cv.line(img,(_2_x1, _2_y1),(_2_x2, _2_y2),(0,0,255),2)
    cv.line(img,(_1_x1, _1_y1),(_1_x2, _1_y2),(0,255,255),2)
    cv.line(img,(_0_x1, _0_y1),(_0_x2, _0_y2),(0,0,255),2)

    if _9_x1 is not 0:
        line_count += 1
        line_list.append([_9_x1,_9_y1,_9_x2,_9_y2])
    if _8_x1 is not 0:
        line_count += 1
        line_list.append([_8_x1,_8_y1,_8_x2,_8_y2])
    if _7_x1 is not 0:
        line_count += 1
        line_list.append([_7_x1,_7_y1,_7_x2,_7_y2])
    if _6_x1 is not 0:
        line_count += 1
        line_list.append([_6_x1,_6_y1,_6_x2,_6_y2])
    if _5_x1 is not 0:
        line_count += 1
        line_list.append([_5_x1,_5_y1,_5_x2,_5_y2])
    if _4_x1 is not 0:
        line_count += 1
        line_list.append([_4_x1,_4_y1,_4_x2,_4_y2])
    if _3_x1 is not 0:
        line_count += 1
        line_list.append([_3_x1,_3_y1,_3_x2,_3_y2])
    if _2_x1 is not 0:
        line_count += 1
        line_list.append([_2_x1,_2_y1,_2_x2,_2_y2])
    if _1_x1 is not 0:
        line_count += 1      
        line_list.append([_1_x1,_1_y1,_1_x2,_1_y2])
    if _0_x1 is not 0:
        line_count += 1    
        line_list.append([_0_x1,_0_y1,_0_x2,_0_y2])

    if line_count <= 2:
        return ["직관",None]
    else :
        for i in range(len(line_list[1:])):  

            a1 = line_list[i][3] - line_list[i][1]
            b1 = line_list[i][0] - line_list[i][2]
            c1 = a1*line_list[i][0] + b1*line_list[i][1]

            a2 = line_list[i+1][3] - line_list[i+1][1]
            b2 = line_list[i+1][0] - line_list[i+1][2]
            c2 = a2*line_list[i+1][0] + b2*line_list[i+1][1]

            deteminate =  a1*b2 - a2*b1
            try:
                t_X = int((b2*c1 - b1*c2)/deteminate)
                t_Y = int((a1*c2 - a2*c1)/deteminate) 
            except ZeroDivisionError as e:
                print(e)

            if abs(t_X) < 1000:
                cv.line(img,(line_list[0][0],line_list[0][1]),(line_list[0][2],line_list[0][3]),(255,0,255),2)
                cv.line(img,(line_list[1][0],line_list[1][1]),(line_list[1][2],line_list[1][3]),(255,0,255),2)
                cv.circle(img, (t_X, t_Y), 7, (0, 0, 255), -1)

                angle_height = math.atan2((line_list[i][0] - t_X),(line_list[i][1] - t_Y))
                angle_bottom = math.atan2((line_list[i+1][2] - t_X),(line_list[i+1][3] - t_Y))
                upper_angle = (angle_height-angle_bottom)*180/math.pi 
                angle_result.append(180-abs(upper_angle))
  

        if sum(angle_result) < 20:
            return ["직관",None]
        else:
            return ["곡관",math.ceil(sum(angle_result))]

