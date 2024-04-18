# -*- coding:utf-8 -*-
"""
作者：知行合一
日期：2019年 06月 18日 15:57
文件名：demo.py
地点：changsha
"""

""""
视频流识别是否佩戴口罩
"""

# 导入相关包
import cv2
import numpy as np
import tensorflow as tf
import pandas
import matplotlib.pyplot as plt
from scipy.special import softmax

class MaskRecognition:
    def __init__(self):
        # 加载模型
        self.model = model = tf.keras.models.load_model('data/face_mask_model/')

        # 打印模型架构
        print(self.model.summary())

    def imgBlob(self,img):
        """
        图片转为Blob格式
        """
        # 转为Blob
        img_blob = cv2.dnn.blobFromImage(img, 1, (100, 100), (104, 177, 123), swapRB=True)
        # 压缩维度
        img_squeeze = np.squeeze(img_blob).T
        # 旋转
        img_rotate = cv2.rotate(img_squeeze, cv2.ROTATE_90_CLOCKWISE)
        # 镜像
        img_flip = cv2.flip(img_rotate, 1)
        # 去除负数并归一化
        img_blob = np.maximum(img_flip, 0) / img_flip.max()

        return img_blob


    def recognize(self):
        """
        识别
        """


        cap = cv2.VideoCapture(0)

        # 获取原图尺寸
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 加载SSD模型
        face_detector = cv2.dnn.readNetFromCaffe('./weights/deploy.prototxt.txt',
                                                 './weights/res10_300x300_ssd_iter_140000.caffemodel')

        # 获取labels
        labels = ['yes','no','nose']

        # 颜色
        colors = [(0,255,0),(0,0,255),(0,255,255)]

        while True:
            ret,frame = cap.read()

            frame = cv2.flip(frame,1)

            # 转为Blob
            img_blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123), swapRB=True)
            # 输入
            face_detector.setInput(img_blob)
            # 推理
            detections = face_detector.forward()
            # 遍历结果

            # 人脸数量
            person_count = detections.shape[2]
            for face_index in range(person_count):
                # 置信度
                confidence = detections[0, 0, face_index, 2]
                # print(confidence)
                if confidence > 0.5:
                    locations = detections[0, 0, face_index, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                    # print(locations)
                    # 取整
                    l, t, r, b = locations.astype('int')

                    # 截取人脸
                    face_crop = frame[t:b,l:r]

                    img_blob = self.imgBlob(face_crop)

                    img_input = img_blob.reshape(1, 100, 100, 3)

                    # 预测
                    result = self.model.predict(img_input)

                    result = softmax(result[0])

                    # 最大值索引
                    max_index = result.argmax()

                    #最大值
                    max_value = result[max_index]

                    # 标签label
                    label = labels[max_index]

                    txt = label + ' ' + str(round(max_value * 100,2)) + '%'

                    color = colors[max_index]

                    # 绘制文字
                    cv2.putText(frame,txt,(l,t-10),cv2.FONT_ITALIC,1,color,2)

                    #   画人脸框
                    cv2.rectangle(frame,(l,t),(r,b),color,5)

                    # cv2.imshow('demo', img_blob)

            cv2.imshow('demo',frame)

            # 退出条件
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# 实例化
mask =  MaskRecognition()
mask.recognize()
