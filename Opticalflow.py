import numpy as np
import cv2 as cv

cap = cv.VideoCapture(cv.samples.findFile("UCF_CrowdsDataset/3687-18_70.mov"))
c=0

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 3 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 0,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

mask = np.zeros_like(old_frame)
#cv.imshow('mask',mask)
while(cap.isOpened()):
    c=c+1
    ret,frame = cap.read()
    if ret == True:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if c%5==0:
            p0 = cv.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 1)
            frame = cv.circle(frame,(int(a),int(b)),2,color[i].tolist(),-1)

        img = cv.add(frame,mask)
        cv.imshow('OpticalFlow',img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            cv.destroyWindow('frame')
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    else:
        break
cv.destroyAllWindows
cv.imshow('of',mask)
cv.waitKey(0)