# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
#code to get the key frame from the video and save it as a png file.

import cv2
import os
#videopath : path of the video file
#frames_path: path of the directory to which the frames are saved
#count: to assign the video order to the frane.
def frameExtractor(videopath,frames_path, subject, lett):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no= int(video_length/2)
    #print("Extracting frame..\n")
    cap.set(1,frame_no)
    ret,frame=cap.read()
    frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(frames_path +"/"+ subject+lett+".png", frame)

# videos = os.listdir("videos/")
videos = ["Sowmya"]
for folder in videos:
    for file in os.listdir("videos/" + folder):
        vidpath = "videos/"+folder+"/"+file
        print(vidpath, "letters",folder,file[:-4][-1])
        frameExtractor(vidpath, "letters",folder,file[:-4][-1])
