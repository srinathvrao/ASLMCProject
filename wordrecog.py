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

import os
# Sowmya - B, H, Q, Z
# Srinath - H
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
				cap.set(3, args.cam_width)
				cap.set(4, args.cam_height)

				start = time.time()
				frame_count = 0
				fps = int(cap.get(cv2.CAP_PROP_FPS))
				frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				duration = frame_length // fps
				tim = 0.5
				timestamps = []
				while tim<duration:
					timestamps.append(tim)
					tim+=1
				timesdone=0
				while True:
					# print(frame_count, cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
					input_image, display_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride)
					check=0
					for timest in timestamps:
						if abs(timest- (cap.get(cv2.CAP_PROP_POS_MSEC)/1000))<0.02:
							print("%.2f %.2f %.2f"%(timest, cap.get(cv2.CAP_PROP_POS_MSEC)/1000,abs(timest - cap.get(cv2.CAP_PROP_POS_MSEC)/1000 )))
							# continue
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
							cv2.imshow('posenet', overlay_image)
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
							    cv2.imshow('wrist_cropped_image'+str(check), resiz)
							    print(wristname)
							    # cv2.waitKey(0)
							    # break
							    # cv2.imwrite("letter_images/"+file+vid[0]+".png",resiz) # break
							# work on display_image
					frame_count += 1
					if cv2.waitKey(1) & 0xFF == ord('q'):
					    break

				print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()