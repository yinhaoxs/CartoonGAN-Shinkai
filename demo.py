# -*- coding: utf-8 -*-
"""
# @Date: 2020-08-14 17:12
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: demo.py
# Copyright @ 2020 yinhao. All rights reserved.
"""

from segment import seg
from style import cartoonize
import numpy as np
import time, os, cv2
import tensorflow as tf



# 寻找最大轮廓，传入的是一个二值的黑白图
def FindBigestContour(src):
    imax = 0
    imaxcontours = -1
    # 返回的是原图片，边界集合，轮廓的属性
    # try:
    image, contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        itemp = cv2.contourArea(contours[i])
        if (imaxcontours < itemp):
            imaxcontours = itemp
            imax = i
    return contours[imax]


# 增强图像的对比度,输入的分别是图的对比度，最后是亮度,返回为转换后图片
def EnhanceSatutrtion(src, alpha, bright):
    blank = np.zeros_like(src, src.dtype)
    dst = cv2.addWeighted(src, alpha, blank, 1-alpha, bright)
    # dst=cv2.addWeighted(src,0.7,blank,0.3,0)#这样才能增加对比度
    return dst


def seamlessClone(src, cloud, mask):
    try:
        ### 融合卡通图与云图
        maxCountour = FindBigestContour(mask)
         # 返回的是（x,y,w,h）四个参数
        maxRect = cv2.boundingRect(maxCountour)
        if maxRect[2]==0 or maxRect[3]==0:
            # maxRect=cv2.RECURS_FILTER
            matDst = src.copy()
            return matDst
        else:
            cloud = cv2.resize(cloud, (maxRect[2], maxRect[3]))
            # print(maxRect[2])
            # print(maxRect[3])

            ## 云图与mask的尺寸相同
            mask_obj = mask[0:maxRect[3], 0:maxRect[2]]

            # 要求为整数，所以传入不能带有小数
            center = (maxRect[2]//2, maxRect[3]//2)

            # Create an all white mask
            # 分别是目标影像，背景影像，目标影响上的mask，
            # 目标影像的中心在背景图像上的坐标！注意是目标影像的中心！
            out_put = cv2.seamlessClone(cloud, src, mask_obj, center, cv2.NORMAL_CLONE)
            temp = cv2.bilateralFilter(out_put.copy(), 5, 10.0, 2.0)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2YCrCb)
            planes = cv2.split(temp)
            planes[0] = cv2.equalizeHist(planes[0])
            temp = cv2.merge(planes)
            temp = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

            matDst=EnhanceSatutrtion(temp, 1.1, 1.3)

        return matDst

    except:
        return src


if __name__ == '__main__':
    t1 = time.time()
    cfg = seg.get_parser()
    cloud_path = 'images/reference.jpg'
    model_path = 'style/checkpoints/'
    image_path = 'images/'
    save_path = 'results/'

    cloud = cv2.imread(cloud_path)
    # 获取模型, 均值, 方差
    model, mean, std, colors = seg.eval()
    num = 0
    for photo_name in os.listdir(image_path):
        num += 1
        t = time.time()
        img_path = os.path.join(image_path+os.sep, photo_name)
        image_photo = cv2.imread(img_path)
        image_seg = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3

        # 生成卡通图
        cartoon_img = cartoonize.cartoonize(image_photo, model_path)
        # 生成mask掩码
        mask_img = seg.seg(model.eval(), image_seg, cfg.classes, mean, std, cfg.base_size, cfg.test_h, cfg.test_w, cfg.scales, colors)
        # 3通道变1通道
        mask_img = cv2.cvtColor(np.array(mask_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        # print(type(mask_img))
        # print(mask_img)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
        matDst = seamlessClone(cartoon_img, cloud, mask_img)

        if matDst is not None:
            cv2.imwrite(save_path+"/"+"{}".format(photo_name), matDst)

        print("单张图片的处理时间:{}".format(time.time()-t))
        print("\n")

    print("单张图片的处理时间:{}".format((time.time()-t1)/num))