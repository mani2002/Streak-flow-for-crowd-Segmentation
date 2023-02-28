import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def watershed(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0,dist_transform.max(),0)
    #0.7*dist_transform.max()
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv.watershed(img,markers)
    img[markers == -1] = [0,255,0]
    return img

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 3 )

lk_params = dict( winSize  = (21,21),
                  maxLevel = 0,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv.VideoCapture(cv.samples.findFile("UCF_CrowdsDataset/2181-2_70.mov"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
p=prvs.copy()
step=16
h, w = prvs.shape[:2]
y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
p0 = cv.goodFeaturesToTrack(prvs, mask = None, **feature_params)
blank=np.zeros(frame1.shape[:3],dtype='uint8')
mask=cv.rectangle(blank,(0,0),(prvs.shape[1],prvs.shape[0]),(255,255,255),-1)
flow=None
mask1= np.zeros_like(frame1)
color = np.random.randint(0,255,(100,3))
c=1
d=0
flow1=None
while(c):
    ret, frame2 = cap.read()
    if ret == True:
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,next, flow, 0.5, 3, 21, 10, 7, 1.1, 0)
        
        
        #optical flow


        p0 = cv.goodFeaturesToTrack(next, mask = None, **feature_params)
        p1, st, err = cv.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p1[st==1]
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask1 = cv.line(mask1, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 1)
            frame2 = cv.circle(frame2,(int(a),int(b)),2,color[i].tolist(),-1)
        img = cv.add(frame2,mask1)
        img=cv.putText(img,'OpticalFlow',(30,30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv.LINE_AA)
        cv.imshow('OpticalFlow',img)
        
        
        #hsv

        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        mask=cv.addWeighted(mask,0.5,bgr,0.5,0)
        mask=cv.putText(mask,'Similarity',(30,30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv.LINE_AA)
        cv.imshow('Similarity',mask)
        
        
        #watershed

        img1=watershed(bgr)
        img1=cv.putText(img1,'WaterShed',(30,30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv.LINE_AA)
        cv.imshow('WaterShed',img1)
        #h2=np.concatenate((mask,img1), axis=1)
        
        
        #Streaklines

        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
        lines = np.int64(lines)
        img_bgr = cv.cvtColor(next, cv.COLOR_GRAY2BGR)
        for (x1, y1), (x2, y2) in lines:
            i=y1%100
            cv.line(img_bgr,(x1,y1),(x2,y2),color[i].tolist(),1)
            cv.circle(img_bgr, (x1, y1), 1,color[i].tolist(), -1)
        if(d==1):
            f1,f2=flow1[fy,fx].T
            lines1=np.vstack([x-fx,y-fy,x-f1,y-f2]).T.reshape(-1, 2, 2)
            lines1 = np.int32(lines1 + 0.5)
            for (x1, y1), (x2, y2) in lines1:
                i=y1%100
                cv.line(img_bgr,(x1,y1),(x2,y2),color[i].tolist(),1)
            

        img_bgr=cv.putText(img_bgr,'StreakLines',(30,30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv.LINE_AA)
        #h1=np.concatenate((img,img_bgr), axis=1)
        #v=np.concatenate((h1,h2),axis=0)
        #cv.imshow('show',v)

        cv.imshow('Streaklines',img_bgr)
        flow1=flow
        d=1
        img_g=cv.cvtColor(img_bgr,cv.COLOR_BGR2GRAY)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
        elif 0xFF == ord('s'):
            cv.imwrite('opticalflow.png',img)
            cv.imwrite('streakline.png',img_bgr)
            cv.imwrite('similarity.png',mask)
            cv.imwrite('watershed.png',img1)
        prvs = next.copy()
    else:
        break