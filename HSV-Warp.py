# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 19:45:27 2018

@author: Rell
"""

import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time
image_hsv = None
pixel = (0,0,0) #RANDOM DEFAULT VALUE

ftypes = [
    ('JPG', '*.jpg *.JPG *.JPEG'), 
    ('PNG', '*.png *.PNG'),
    ('all files' ,'.*')
]

def pick_color(event,x,y,flags,param):
    global upper,lower
    if event == cv.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        upper =  np.array((pixel[0] + 20, pixel[1] + 20, pixel[2] + 100))
        lower =  np.array((pixel[0] - 20, pixel[1] - 20, pixel[2] - 100))
        print('('+str(lower[0])+','+str(lower[1])+','+str(lower[2])+'),('+str(upper[0])+','+str(upper[1])+','+str(upper[2])+')')

        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 
        image_mask = cv.inRange(image_hsv,lower,upper)
        cv.imshow("Mask",image_mask)

def main():

    global image_hsv, pixel

    #OPEN DIALOG FOR READING THE IMAGE FILE
    root = tk.Tk()
    root.withdraw() #HIDE THE TKINTER GUI
    # file_path = filedialog.askopenfilename(filetypes = ftypes,title='Title')
    file_path = filedialog.askopenfilename(filetypes = ftypes,title='Title')
    image_src = cv.imread(file_path)
    cv.imshow("BGR",image_src)

    #CREATE THE HSV FROM THE BGR IMAGE
    image_hsv = cv.cvtColor(image_src,cv.COLOR_BGR2HSV)
    cv.imshow("HSV",image_hsv)

    #CALLBACK FUNCTION
    cv.setMouseCallback("HSV", pick_color)

    cv.waitKey(0)
    cv.destroyAllWindows()

    start = time.time()
    ocr(file_path,upper,lower)
    print(time.time()-start)

def ocr(filepath,upper,lower):
    def intersection(line1, line2):
        """
        Finds the intersection of two lines given in Hesse normal form.
    
        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]


    def segmented_intersections(lines):
        """Finds the intersections between groups of lines."""
    
        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i+1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(intersection(line1, line2)) 
    
        return intersections
    from collections import defaultdict
    def segment_by_angle_kmeans(lines, k=2, **kwargs):
        """Groups lines based on angle with k-means.
    
        Uses k-means on the coordinates of the angle on the unit circle 
        to segment `k` angles inside `lines`.
        """
    
        # Define criteria = (type, max_iter, epsilon)
        default_criteria_type = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
        flags = kwargs.get('flags', cv.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get('attempts', 10)
    
        # returns angles in [0, pi] in radians
        angles = np.array([line[0][1] for line in lines])
        # multiply the angles by two and find coordinates of that angle
        pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                        for angle in angles], dtype=np.float32)
    
        # run kmeans on the coords
        labels, centers = cv.kmeans(pts, k, None, criteria, attempts, flags)[1:]
        labels = labels.reshape(-1)  # transpose to row vec
    
        # segment lines based on their kmeans label
        segmented = defaultdict(list)
        for i, line in zip(range(len(lines)), lines):
            segmented[labels[i]].append(line)
        segmented = list(segmented.values())
        return segmented

    import math
    
    ## Read
    img = cv.imread(filepath)
    
    #Depends if need to resize smaller
#    img = cv.resize(img, (0,0), fx=0.2, fy=0.2)
    
    ## convert to hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    ## slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    
    #====dilation
    kernel = np.ones((5,5),np.uint8)
    #erosion = cv.erode(green,kernel,iterations = 1)
    dil = cv.dilate(green,kernel,iterations = 1)
    
    #======Hough
    
    
    #img = cv.imread(r'C:/Users/Rell/Downloads/Cynopsis/junyang.jpg',0)
    #src = cv.resize(img, (0,0), fx=0.2, fy=0.2)
    src = dil.copy()
    #not sure whats the 3rd parameter but seems good to play around with
    dst = cv.Canny(src, 100, 300, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    
    #4th parameter threshold number of points to define as a line. Default value: 100
    lines = cv.HoughLines(dst, 1, np.pi / 180, 110, None, 0, 0)

    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
#    cv.imwrite("image_lines.jpg",cdst)
    
    #=========All intersections
    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented)
    
    #===============Extreme four corners
    temp = []
    for i in intersections:
        i=i[0]
        temp.append(i)
    
    import heapq
    
    ycord1 =[]
    pos_ycord1 = heapq.nlargest(int(len(temp)/1.2), temp,key=lambda x: x[0])
    for k,i in enumerate(pos_ycord1):
        if k+1 == int(len(temp)/1.5):
            break
        if abs(i[0]-pos_ycord1[k+1][0]) > 100: 
            ycord1.append(i)
            break
        else:
            ycord1.append(i)
    br = max(ycord1,key=lambda x: x[1])#this is supposed bottom right
    tr = min(ycord1,key=lambda x: x[1])#this is top right
    
    ycord2=[]
    pos_ycord2 = heapq.nsmallest(int(len(temp)/1.5), temp,key=lambda x: x[0])
    for k,i in enumerate(pos_ycord2):
        if k+1 == int(len(temp)/1.5):
            break
        if abs(i[0]-pos_ycord2[k+1][0]) > 100: 
            ycord2.append(i)
            break
        else:
            ycord2.append(i)
    bl = max(ycord2,key=lambda x: x[1])#this is bottom left
    tl = min(ycord2,key=lambda x: x[1])#this is top left
    
    #=========warping
    pts1 = np.float32([tl,tr,bl,br])
    pts2 = np.float32([[0,0],[600,0],[0,400],[600,400]])
    
    M = cv.getPerspectiveTransform(pts1,pts2)
    warped = cv.warpPerspective(img,M,(600,400))
    cv.imshow("Warped",warped)
    cv.waitKey(0)
    cv.destroyAllWindows()
#    cv.imwrite('warped_img2.jpg', warped)


if __name__=='__main__':
    main()
