import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

# ========================================
# ================ CAMShift ==============
# ========================================
def skeleton_tracker_CamShift(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, c+w/2, r+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you


    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # perform the tracking
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        '''
        # Draw it on image###use this block to verify results visually
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.circle(frame,(int(c + w/2.0),int(r + h/2.0)), 4, (255,255,255), -1)

        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        '''
        c, r, w, h = track_window
        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter, c+w/2, r+h/2)) # Write the ecntre of the rectangle
        frameCounter = frameCounter + 1

    output.close()

# ===============================================
# ================ Particle Filter ==============
# ===============================================
def skeleton_tracker_PF(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    
    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return
    
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    n_particles = 200
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, c + w/2.0,r + h/2.0)) # Write as frame_index,pt_x,pt_y(weighted average)
    frameCounter = frameCounter + 1
    # calculate the HSV histogram in the window
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
    f0 = particleevaluator(hist_bp, init_pos) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # perform the tracking
        stepsize = 8
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1], frame.shape[0]))-1).astype(int)
        '''
        for i in range(0, len(particles)-1):
         img1 = cv2.circle(frame,(particles[i][0],particles[i][1]), 2, (255,255,255), -1)
         cv2.imshow('img2',img1)
        '''
        f   = particleevaluator(hist_bp, particles.T) # Evaluate particles
        ##weights  = 1./(1. + (f0-f)**2)
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
        
        '''
        # Draw it on image
        img2 = cv2.circle(frame,(pos[0],pos[1]), 5, (0,255,0), -1)
        cv2.imshow('img2',img2)
        cv2.waitKey(60)
        '''
        output.write("%d,%d,%d\n" % (frameCounter, pos[0], pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights
        

    output.close()

# =============================================================
# ================ Face Detector + Kalman Filter ==============
# =============================================================
def skeleton_tracker_KF(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    
    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, c + w/2.0,r + h/2.0))
    frameCounter = frameCounter + 1

    # initialize the tracker

    kalman = cv2.KalmanFilter(4,2,0)
    '''
        For Kalman Filter:
        
        # --- init
        '''
    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        # perform the tracking
        prediction = kalman.predict()

        # obtain measurement
        c, r, w, h = detect_one_face(frame)
        
        if cmp((c,w,r,h), (0,0,0,0))!=0:# e.g. face found
            ##use this block to visualize the face detector result
            '''
            cv2.circle(frame,(int(c+w/2),int(r+h/2)), 4, (255,0,0), -1)
            '''
            measurement = np.array([c+w/2, r+h/2], dtype='float64')
            posterior = kalman.correct(measurement)
            kalman.statePost = posterior
        else:
            kalman.statePost = prediction
        
        ##use this block to visualize the prediction and posterior
        '''
        cv2.circle(frame,(int(prediction[0]),int(prediction[1])), 6, (0,0,255), -1)
        cv2.circle(frame,(int(kalman.statePost[0]),int(kalman.statePost[1])), 4, (0,255,0), -1)
        cv2.imshow('img2',frame)
        cv2.waitKey(60)
        '''
        # write the result to the output file
        #use below float representation if you want to track exact positions, which will also show that values are chnaging but slowly
        #output.write("%d,%.3f,%.3f\n" % (frameCounter, kalman.statePost[0], kalman.statePost[1])) # Write as frame_index,pt_x,pt_y
        # use prediction or posterior as your tracking result
        output.write("%d,%d,%d\n" % (frameCounter, kalman.statePost[0], kalman.statePost[1])) # rounding off position coordinates as int positions
        frameCounter = frameCounter + 1
            
    output.close()

# ==========================================================================
# ================ Bonus: Face Detector + Optical Flow tracker==============
# ==========================================================================

def skeleton_tracker_OF(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")
    
    
    lk_params = dict( winSize  = (15, 15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
    feature_params = dict( maxCorners = 500,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7 )
 
 
    _ret, frame =  v.read()
    frameCounter = 0
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, c + w/2.0,r + h/2.0))
    frameCounter = frameCounter + 1

    track_len = 10
    detect_interval = 5##used to reconsider good features
    tracks = []
    frame_idx = 0
    
    while True:
       _ret, frame =  v.read()
       if _ret == False:
           return

       frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       vis = frame.copy()
       sumx = 0 #to calculate avg of x coordinates
       sumy = 0 #to calculate avg of y coordinates
       count = 0
    
       c,r,w,h = detect_one_face(frame)
       #if c!=0 and r!=0 and w!=0 and h!=0:
       if cmp((c,w,r,h), (0,0,0,0))!=0:#i.e. face detector succeeds
           '''
           cv2.circle(vis, (int(c + w/2), int(r + h/2)), 5, (255, 0, 0), -1)
           '''
           output.write("%d,%d,%d\n" % (frameCounter, c + w/2.0,r + h/2.0)) # Write the centre of the rectangle detected by face detector

        ##we will use OF whenever face detector fails to find a face
       if len( tracks) > 0:
           img0, img1 =  prev_gray, frame_gray
           p0 = np.float32([tr[-1] for tr in  tracks]).reshape(-1, 1, 2)
           p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
           p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
           d = abs(p0-p0r).reshape(-1, 2).max(-1)
           good = d < 1
           new_tracks = []
           for tr, (x, y), good_flag in zip( tracks, p1.reshape(-1, 2), good):
               if not good_flag:
                   continue
               tr.append((x, y))
               if len(tr) >  track_len:
                   del tr[0]
               new_tracks.append(tr)
               sumx = sumx + x
               sumy = sumy + y
               count = count + 1
               '''
               cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
               '''
           tracks = new_tracks
       
       ###handle drifting
       if  frame_idx %  detect_interval == 0:
           mask = np.zeros_like(frame_gray)
           mask[r:r+h, c:c+w] = 255
           for x, y in [np.int32(tr[-1]) for tr in  tracks]:
               '''
               cv2.circle(mask, (x, y), 5, 0, -1)
               '''
               sumx = sumx + x
               sumy = sumy + y
               count = count + 1
           p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
           if p is not None:
               for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])
                                                                           
                                                                           
       frame_idx += 1
       prev_gray = frame_gray
       if cmp((c,w,r,h), (0,0,0,0))==0:#if face detector fails
           '''
           cv2.circle(vis, (int(sumx/count), int(sumy/count)), 5, (0, 0, 255), -1)
           cv2.polylines(vis, [np.int32(tr) for tr in  tracks], False, (0, 255, 0))#track movement of face
           '''
           output.write("%d,%d,%d\n" % (frameCounter, int(sumx/count), int(sumy/count))) # Write the mean of all the good points detected by OF
       ##use this block to visualize the result
       '''
       cv2.imshow('lk_track', vis)
       ch = cv2.waitKey(60)
       if ch == 27:
           break
       '''

       frameCounter = frameCounter + 1
   
    output.close()

'''
    Main Function
    Usage:
    CamShift: python detection_tracking.py 1 02-1.avi ./
    Particle Filter: python detection_tracking.py 2 02-1.avi ./
    Face detector + Kalman Filter: python detection_tracking.py 3 02-1.avi ./
    Face Detector + Optical Flow tracker: python detection_tracking.py 4 02-1.avi ./
    
    '''
if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        skeleton_tracker_CamShift(video, "output_camshift.txt")
    elif (question_number == 2):
        skeleton_tracker_PF(video, "output_particle.txt")
    elif (question_number == 3):
        skeleton_tracker_KF(video, "output_kalman.txt")
    elif (question_number == 4):
        skeleton_tracker_OF(video, "output_of.txt")


