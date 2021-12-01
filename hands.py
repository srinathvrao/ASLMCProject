import cv2
import mediapipe as mp
import time
import numpy as np


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
import os
folders = os.listdir("word_images")
for folder in ["Sowmya"]:
	imnames = os.listdir("word_images/"+folder)
	for imname in imnames:
		# print("word_images/"+folder+"/"+imname)
		img = cv2.imread("word_images/"+folder+"/"+imname)
		# success, img = cap.read()
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = hands.process(imgRGB)
		#print(results.multi_hand_landmarks)
		im = np.array([0])
		if results.multi_hand_landmarks:
		    for handLms in results.multi_hand_landmarks:
		        xpoints = []
		        ypoints = []
		        for id, lm in enumerate(handLms.landmark):
		            #print(id,lm)
		            h, w, c = img.shape
		            cx, cy = int(lm.x *w), int(lm.y*h)
		            #if id ==0:
		            xpoints.append(cx)
		            ypoints.append(cy)
		            # cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
		        xpoints = sum(xpoints)//len(xpoints)
		        ypoints = sum(ypoints) // len(ypoints)
		        startx,starty=0,0
		        if xpoints<100:
		        	startx=0
		        else:
		        	startx = xpoints-100
		        endx = xpoints+100
		        if ypoints<100:
		        	starty=0
		        else:
		        	starty = ypoints-100
		        endy = ypoints+100
		        im = img[starty:endy, startx:endx]
		        im = cv2.resize(im, (224,224),interpolation=cv2.INTER_AREA)
		        cv2.imwrite("word_images/"+folder+"/"+imname,im)
		        # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
		else:
			print("no landmarks",folder,imname)

		cTime = time.time()
		fps = 1/(cTime-pTime)
		pTime = cTime

		cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

		cv2.imshow("Image", img)
		if len(im)!=1:
			cv2.imshow("hand",im)
		# cv2.waitKey(0)

