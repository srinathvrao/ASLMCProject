import cv2
import mediapipe as mp
import time
import numpy as np

import os
folders = os.listdir("word_images")
for folder in folders:
	imnames = os.listdir("word_images/"+folder)
	for imname in imnames:
		# print("word_images/"+folder+"/"+imname)
		img = cv2.imread("word_images/"+folder+"/"+imname)
		if img.shape[0]<250:
			img = cv2.resize(img, (224,224),interpolation=cv2.INTER_AREA)
			cv2.imwrite("word_images/"+folder+"/"+imname,img)
		else:
			print(folder,imname)