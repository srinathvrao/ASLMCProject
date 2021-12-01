import tensorflow as tf
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default="test_words", help="Optionally use a video file instead of a live camera")
args = parser.parse_args()
import matplotlib.pyplot as plt
import os
import numpy as np
# Sowmya - B, H, Q, Z
# Srinath - H
from scipy.signal import savgol_filter

def main():
	with tf.Session() as sess:
		model_cfg, model_outputs = posenet.load_model(args.model, sess)
		output_stride = model_cfg['output_stride']
		folders = os.listdir(args.file)
		for file in ["Sowmya"]:
			vidpath = args.file+"/"+file
			vids = os.listdir(vidpath)
			for vid in vids:
				vidpath = args.file+"/"+file+"/"+vid
				cap = cv2.VideoCapture(vidpath)
				print(vidpath)
				cap.set(3, args.cam_width)
				cap.set(4, args.cam_height)

				start = time.time()
				frame_count = 0
				frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				input_image, prev_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride)
				prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
				pixsum = []
				framecs = []
				ims = []
				while frame_count<frame_length-1:
					frame_count += 1
					input_image, display_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride)
					# if frame_count%2==0 or frame_count%3==0:
					# 	pass
					# else:
						# print(frame_count)
					dispim = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
					backsub_image = cv2.subtract(dispim,prev_image)
					cv2.imshow("backsub",backsub_image)
					sss = np.sum(backsub_image)
					# if sss <= 400000:
					pixsum.append(sss)
					framecs.append(frame_count)
					ims.append(display_image)
					# cv2.imshow("original",display_image)
					prev_image = dispim
					if cv2.waitKey(1) & 0xFF == ord('q'):
					    break

				# plt.figure()
				margin = 5
				plen = len(pixsum)
				minframes = []
				pixsums = []
				imframes = []
				for i, pixs in enumerate(pixsum):
					if i>margin:
						if i<plen-margin:
							check=0
							for k in range(i-margin,i+margin):
								if pixs < pixsum[k]:
									check+=1
							if check==2*margin-1:
								minframes.append(framecs[i])
								pixsums.append(pixs)
						else:
							check=0
							for k in range(i-margin,plen):
								if pixs < pixsum[k]:
									check+=1
							if check==plen-i+margin-1:
								minframes.append(framecs[i])
								pixsums.append(pixs)
					else:
						check=0
						for k in range(i+margin):
							if pixs < pixsum[k]:
								check+=1
						if check==i+margin-1:
							minframes.append(framecs[i])
							pixsums.append(pixs)
				print(pixsums)
				print(minframes, len(ims))
				frameslen = len(minframes)
				reducedframes = []
				i=0
				while i<frameslen-1:
					minpixs = pixsums[i]
					redfr = minframes[i]
					k=i+1
					while k<frameslen and minframes[k] - minframes[i] <= 45:
						# print(k,i,minframes[k] - minframes[i])
						if pixsums[k]<minpixs:
							redfr = minframes[k]
							minpixs = pixsums[k]
						k+=1
					reducedframes.append(redfr)
					imframes.append(ims[redfr-1])
					i=k
				print(reducedframes)
				ccz=0
				for imfr in imframes:
					print(imfr.shape)
					cv2.imwrite("word_images/"+file+"/"+vid.split(".")[0]+str(ccz)+".png",imfr)
					ccz+=1
				# yhat = savgol_filter(pixsum, 15, 2)
				# print(np.argsort(pixsum[15:])[:5])
				# plt.figure()
				# plt.plot(framecs, pixsum,'green')
				# plt.plot(framecs, yhat,'red')
				# plt.show()
				print(frame_count, vidpath)
				print('Average FPS: ', frame_count / (time.time() - start))
				

if __name__ == "__main__":
    main()