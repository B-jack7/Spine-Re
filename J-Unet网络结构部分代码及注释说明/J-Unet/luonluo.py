import numpy as np
import cv2
import os


imgs = os.listdir('./data/test/')
for jpg in imgs:
    imbgr = cv2.imread('./data/test/'+jpg)
    imgray = cv2.imread('./data/test_labels/'+jpg,0)
    imgray1 = cv2.imread('./result_pics/'+jpg,0)
        # 以灰度图读取mask图片，那么不同物体对应的像素值是不同的，提取统一像素值的所有位置

        # mask中物体1的像素值为78，设定像素范围来获取所有坐标
    upper1=imgray<=255
    lower1=imgray>254
    thresh1=(np.multiply(upper1,lower1).astype(np.float32)*255).astype(np.uint8)
    upper1=imgray<=100
    lower1=imgray>0
    thresh2=(np.multiply(upper1,lower1).astype(np.float32)*255).astype(np.uint8)
    upper1=imgray1<=100
    lower1=imgray1>0
    thresh3=(np.multiply(upper1,lower1).astype(np.float32)*255).astype(np.uint8)
    upper1=imgray1<=255
    lower1=imgray1>254
    thresh4=(np.multiply(upper1,lower1).astype(np.float32)*255).astype(np.uint8)
        # # mask中物体2的像素值为35，设定像素范围来获取所有坐标
        # upper2=imgray<=2
        # lower2=imgray>1
        # thresh2=(np.multiply(upper2,lower2).astype(np.float32)*255).astype(np.uint8)

    contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy2 = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours4, hierarchy2 = cv2.findContours(thresh4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(imbgr, contours1, -1, (0,0,255), 1)
    cv2.drawContours(imbgr, contours2, -1, (0,255,0), 1)
    cv2.drawContours(imbgr, contours3, -1, (255,0,255), 1)
    cv2.drawContours(imbgr, contours4, -1, (0,255,255), 1)
    savepath='./vis/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        # cv2.drawContours(imbgr, contours2, -1, (0,255,255), 1)
    cv2.imwrite(savepath+'/'+jpg,imbgr)
