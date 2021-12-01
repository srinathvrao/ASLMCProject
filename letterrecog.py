import tensorflow as tf
import cv2
import time
import argparse
import numpy as np

keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model
import posenet

import os
# Sowmya - B, H, Q, Z
# Srinath - H
def main():
	folders = os.listdir("word_images")
	modelops = []
	actualops = []
	model = load_model('modelt.h5')
	words = []
	for folder in folders:
		imnames = os.listdir("word_images/"+folder)
		for imname in imnames:
			print(folder,imname)
			# print("word_images/"+folder+"/"+imname)
			img = cv2.imread("word_images/"+folder+"/"+imname)
			# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (200, 200))
			# img_arr = np.array(img) / 255.0
			img_arr = img.reshape(1, 200, 200, 3)
			modelops.append(model.predict(img_arr).argmax(1)[0])
			actualops.append(imname[int(imname[-5])].upper())
			words.append(imname[:-5])
	correct, total=0,0
	corrwords = []
	for i,kk in enumerate(actualops):
		if modelops[i] == ord(kk)-65:
			correct+=1
		corrwords.append([words[i],chr(modelops[i]+65)])
			# cv2.imshow('correct',cropped_images[i])
			# cv2.waitKey(0)
		total+=1
	print(correct, total)
	print("Accuracy: %.2f"%(correct/total))
	dwo = {}
	for wor,alph in corrwords:
		if wor not in dwo:
			dwo[wor] = [alph]
		else:
			dwo[wor].append(alph)
	print(dwo)


if __name__ == "__main__":
    main()

'''
34 165
Accuracy: 0.21
{'FUND': 1, 'LOW': 1, 'TRUST': 2, 'FRAME': 1, 'HARSH': 1, 'LANE': 2, 'LEASE': 1, 'LUNG': 2, 'VIDEO': 2, 'CAR': 1, 'Glare': 1, 'Monk': 1, 'Mouse': 2, 'Say': 1, 'Trail': 1, 'Wash': 1, 'agent': 1, 'fate': 1, 'fun': 1, 'iron': 2, 'lamb': 1, 'mild': 1, 'sail': 2, 'throw': 2, 'war': 2}
'''