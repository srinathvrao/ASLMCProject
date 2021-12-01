import tensorflow as tf
import cv2
import time
import argparse
import numpy as np

keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model
import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default="word_images", help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

import os
# Sowmya - B, H, Q, Z
# Srinath - H
def main():
	with tf.Session() as sess:
		model_cfg, model_outputs = posenet.load_model(args.model, sess)
		output_stride = model_cfg['output_stride']
		folders = os.listdir(args.file)
		cropped_images = []
		imnames = []
		for file in folders:
			impath = args.file+"/"+file
			ims = os.listdir(impath)
			for im in ims:
				cap = args.file+"/"+file+"/"+im
				print(cap)
				check=0
				# print(frame_count, cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
				input_image, display_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride)
				heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
				    model_outputs,
				    feed_dict={'image:0': input_image}
				)
				pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
				    heatmaps_result.squeeze(axis=0),
				    offsets_result.squeeze(axis=0),
				    displacement_fwd_result.squeeze(axis=0),
				    displacement_bwd_result.squeeze(axis=0),
				    output_stride=output_stride,
				    max_pose_detections=10,
				    min_pose_score=0.15)

				keypoint_coords *= output_scale

				# TODO this isn't particularly fast, use GL for drawing and display someday...
				overlay_image, wrist, wristname = posenet.draw_skel_and_kp(
				    display_image, pose_scores, keypoint_scores, keypoint_coords,
				    min_pose_score=0.15, min_part_score=0.1)
				# cv2.imshow('posenet', overlay_image)
				if len(wrist) != 0:
				    wristx, wristy = int(wrist[0]), int(wrist[1])
				    startx, starty, endx, endy=0,0,0,0
				    nbyn = 200
				    if wristx<nbyn:
				    	startx=0
				    	endx=wristx+nbyn
				    	starty=wristy- int(1.3*nbyn)
				    	endy=wristy+ int(1.3*nbyn)
				    else:
				    	startx=wristx-nbyn
				    	endx=wristx+nbyn
				    	starty= wristy - int(1.3*nbyn)
				    	endy= wristy + int(1.3*nbyn)
				    resiz = cv2.resize(display_image[starty:endy ,startx:endx], (224,224), interpolation=cv2.INTER_AREA)
				    resiz = cv2.cvtColor(resiz, cv2.COLOR_BGR2GRAY)
				    # cv2.imshow('wrist_cropped_image'+str(check), resiz)
				    cropped_images.append(resiz)
				    imnames.append(im)
				    # break
				else:
					yc = display_image.shape[1] // 2
					xc = display_image.shape[0] // 2
					resiz = cv2.resize(display_image[yc - 112:yc + 111 ,xc-112:xc+111], (224,224), interpolation=cv2.INTER_AREA)
					resiz = cv2.cvtColor(resiz, cv2.COLOR_BGR2GRAY)
					cropped_images.append(resiz)
					imnames.append(im)
				    # cv2.imwrite("letter_images/"+file+vid[0]+".png",resiz) # break
				# work on display_image
		cropped_images = np.array(cropped_images)
		model = load_model('model2.h5')
		print(cropped_images.shape)
		imnamelist = [x[:-4][:-1] for x in imnames]
		imnameids = [int(x[:-4][-1]) for x in imnames]
		actual_op = []
		for i,idd in enumerate(imnameids):
			if idd>=len(imnamelist[i]):
				actual_op.append(ord(imnamelist[i][-1])-65)
			else:
				actual_op.append(ord(imnamelist[i][idd])-65)
		imops = []
		for cr in cropped_images:
			img = cv2.resize(cr, (200, 200))
			img_arr = np.array(img) / 255.0
			img_arr = img_arr.reshape(1, 200, 200, 1)
			imops.append(model.predict(img_arr).argmax(1))
		print(len(actual_op))
		print(len(imops))
		correct, total=0,0
		for i,kk in enumerate(actual_op):
			if imops[i] == kk:
				correct+=1
				# cv2.imshow('correct',cropped_images[i])
				# cv2.waitKey(0)
			total+=1
		print(correct, total)
		print("Accuracy: %.2f"%(correct/total))



if __name__ == "__main__":
    main()