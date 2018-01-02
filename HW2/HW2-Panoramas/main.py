# Instructions:
# Do not change the output file names, use the helper functions as you see fit

import os
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

'''
Laplacian Blending Function for blending stitched images
'''

def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]
    
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))
    
    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in xrange(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
       	gpAsize = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i], dstsize = gpAsize))

        
        gpBsize = (gpB[i-1].shape[1], gpB[i-1].shape[0])
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i], dstsize = gpBsize))

        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks
 
    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)
    
    # now reconstruct
    ls_ = LS[0]
    
    for i in xrange(1,num_levels):
	size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize = size)
	ls_ = cv2.add(ls_, LS[i])

    
    return ls_

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Perspective warping")
   print("2 Cylindrical warping")
   print("3 Bonus perspective warping")
   print("4 Bonus cylindrical warping")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image1] " + "[path to input image2] " + "[path to input image3] " + "[output directory]")

'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)

    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''
def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in xrange(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        #plt.figure(),plt.xticks([]),plt.yticks([])
        #plt.imshow(img3,)
	#plt.show()
        #plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
	#plt.show()
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl,cyl_mask)

'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)

    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''
def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst,True)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        #M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)
   
# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, img3):
	# first, make some room around img1

	img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)	
	(M, pts1, pts2, mask) = getTransform(img2, img1,'homography')

    	# then transform img2 with the 3x3 transformation matrix
    	out1 = cv2.warpPerspective(img2, M, (img1.shape[1],img1.shape[0]), dst=img1.copy(), borderMode=cv2.BORDER_TRANSPARENT)
	#plt.imshow(out1, cmap='gray')
        #plt.show()
	
	(M2, pts12, pts22, mask2) = getTransform(img3,out1,'homography')
        # then transform img3 with the 3x3 transformation matrix
	final = cv2.warpPerspective(img3, M2, (out1.shape[1],out1.shape[0]), dst=out1, borderMode=cv2.BORDER_TRANSPARENT)
	output_image = final
	#plt.imshow(output_image, cmap='gray')
        #plt.show()
	#print final.shape

	# Write out the result
	output_name = sys.argv[5] + "output_homography.png"
	cv2.imwrite(output_name, output_image)
	
	return True

# ===================================================
# =========Blending on Perspective Warping ==========
# ===================================================

def Bonus_perspective_warping(img1, img2, img3):
	# first, make some room around img1
        img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
        (M, pts1, pts2, mask) = getTransform(img2, img1,'homography')

        # then transform img2 with the 3x3 transformation matrix
        out1 = cv2.warpPerspective(img2, M, (img1.shape[1],img1.shape[0]), dst=img1.copy(), borderMode=cv2.BORDER_TRANSPARENT)
        #plt.imshow(out1, cmap='gray')
        #plt.show()

        (M2, pts12, pts22, mask2) = getTransform(img3,img1,'homography')
        # then transform img3 with the 3x3 transformation matrix and stitching warped img3 on img1 to the resultant of img2 warped on image1

        out2 = cv2.warpPerspective(img3, M2, (img1.shape[1],img1.shape[0]), dst=img1.copy(), borderMode=cv2.BORDER_TRANSPARENT)
        #plt.imshow(out2, cmap='gray')
        #plt.show()
        #print out2.shape
	
	#create a mask
	m = np.zeros_like(img1, dtype='float32')
	m[:,img1.shape[1]/2:] = 1 # make the mask half-and-half
        out = Laplacian_Pyramid_Blending_with_mask(out1, out2, m, 5)
    
        #plt.imshow(out, cmap='gray')
        #plt.show()
        
        output_image = out # This is output
        # Write out the result
        output_name = sys.argv[5] + "output_homography_lpb.png"
        cv2.imwrite(output_name, output_image)
        return True	

	
# ===================================================
# =============== Cynlindrical Warping ==============
# ===================================================
def Cylindrical_warping(img1, img2, img3):
    
    # Write your codes here
    
    h1,w1 = img1.shape
    f1 = 415
    K1 = np.array([[f1, 0, w1/2], [0, f1, h1/2], [0, 0, 1]]) # mock calibration matrix
    img1cyl, img1mask = cylindricalWarpImage(img1, K1)
    
    h2,w2 = img2.shape
    f2 = 415
    K2 = np.array([[f2, 0, w2/2], [0, f2, h2/2], [0, 0, 1]]) # mock calibration matrix
    img2cyl, img2mask = cylindricalWarpImage(img2, K2)
    
    h3,w3 = img3.shape
    f3 = 415
    K3 = np.array([[f3, 0, w3/2], [0, f3, h3/2], [0, 0, 1]]) # mock calibration matrix
    img3cyl, img3mask = cylindricalWarpImage(img3, K3)
    
    img1cyl = cv2.copyMakeBorder(img1cyl, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
    #==========img1cyl and img2cyl on the plane of img1cyl===============
    (M1, pts1, pts2, mask1) = getTransform(img2cyl, img1cyl, method = 'affine')
 
    out1_mask = cv2.warpAffine(img2mask, M1[:2,:], (img1cyl.shape[1],img1cyl.shape[0]), dst=np.zeros_like(img1cyl), borderMode=cv2.BORDER_TRANSPARENT)
    out1 = cv2.warpAffine(img2cyl, M1[:2,:], (img1cyl.shape[1],img1cyl.shape[0]), dst=img1cyl.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    ls = out1*(out1_mask/255)+img1cyl*(1-out1_mask/255)
    #plt.imshow(ls, cmap='gray')
    #plt.show()

    #==========out and img3cyl on the plane of out===============
    (M2, pts1, pts2, mask2) = getTransform(img3cyl, ls, method = 'affine')
    # transform img3 with the 3x3 transformation matrix
    out2_mask = cv2.warpAffine(img3mask, M2[:2,:], (ls.shape[1],ls.shape[0]), dst=np.zeros_like(ls), borderMode=cv2.BORDER_TRANSPARENT)#using mask to get rid of black border
    out2 = cv2.warpAffine(img3cyl, M2[:2,:], (ls.shape[1],ls.shape[0]), dst=ls.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    ls = out2*(out2_mask/255)+ls*(1-out2_mask/255)
    #plt.imshow(ls, cmap='gray')
    #plt.show()
    
    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical.png"
    cv2.imwrite(output_name, ls)
       
    return True


# ===================================================
# =========Blending on Cylindrical  Warping ==========
# ===================================================
def Bonus_cylindrical_warping(img1, img2, img3):

    h1,w1 = img1.shape
    f1 = 415
    K1 = np.array([[f1, 0, w1/2], [0, f1, h1/2], [0, 0, 1]]) # mock calibration matrix
    img1cyl, img1mask = cylindricalWarpImage(img1, K1)

    h2,w2 = img2.shape
    f2 = 415
    K2 = np.array([[f2, 0, w2/2], [0, f2, h2/2], [0, 0, 1]]) # mock calibration matrix
    img2cyl, img2mask = cylindricalWarpImage(img2, K2)

    h3,w3 = img3.shape
    f3 = 415
    K3 = np.array([[f3, 0, w3/2], [0, f3, h3/2], [0, 0, 1]]) # mock calibration matrix
    img3cyl, img3mask = cylindricalWarpImage(img3, K3)

    img1cyl = cv2.copyMakeBorder(img1cyl, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
    #==========img1 and img2 on the plane of img1===============
    (M1, pts1, pts2, mask1) = getTransform(img2cyl, img1cyl, method = 'affine')

    out1_mask = cv2.warpAffine(img2mask, M1[:2,:], (img1cyl.shape[1],img1cyl.shape[0]), dst=np.zeros_like(img1cyl), borderMode=cv2.BORDER_TRANSPARENT)
    out1 = cv2.warpAffine(img2cyl, M1[:2,:], (img1cyl.shape[1],img1cyl.shape[0]), dst=img1cyl.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    ls1 = out1*(out1_mask/255)+img1cyl*(1-out1_mask/255)
    #plt.imshow(ls1, cmap='gray')
    #plt.show()

    #==========out and img3 on the plane of out===============
    (M2, pts1, pts2, mask2) = getTransform(img3cyl,img1cyl, method = 'affine')
    # transform img3 with the 3x3 transformation matrix
    out2_mask = cv2.warpAffine(img3mask, M2[:2,:], (img1cyl.shape[1],img1cyl.shape[0]), dst=np.zeros_like(img1cyl), borderMode=cv2.BORDER_TRANSPARENT)
    out2 = cv2.warpAffine(img3cyl, M2[:2,:], (img1cyl.shape[1],img1cyl.shape[0]), dst=img1cyl.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    ls2 = out2*(out2_mask/255)+img1cyl*(1-out2_mask/255)
    #plt.imshow(ls2, cmap='gray')
    #plt.show()

    m = np.zeros_like(img1cyl, dtype='float32')
    m[:,img1cyl.shape[1]/2:] = 1 # make the mask half-and-half
       
    out11 = Laplacian_Pyramid_Blending_with_mask(ls1,ls2, m)
    output_image = out11 # This is our output
	
    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical_lpb.png"
    cv2.imwrite(output_name, output_image)
	
    return True
	
'''
This exact function will be used to evaluate your results for HW2
Compare your result with master image and get the difference, the grading
criteria is posted on Piazza
'''

def RMSD(questionID, target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:
        nonZero_target = cv2.countNonZero(target)
        nonZero_master = cv2.countNonZero(master)

        if (questionID == 1):
           if (nonZero_target < 1200000):
               return -1
        elif(questionID == 2):
            if (nonZero_target < 700000):
                return -1
        else:
            return -1

        total_diff = 0.0;
        master_channels = cv2.split(master);
        target_channels = cv2.split(target);

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1/2.0)
        return total_diff;


if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) != 6):
      help_message()
      sys.exit()
   else: 
      question_number = int(sys.argv[1])
      if (question_number > 4 or question_number < 1):
	 print("Input parameters out of bound ...")
         sys.exit()
		 
   input_image1 = cv2.imread(sys.argv[2], 0)
   input_image2 = cv2.imread(sys.argv[3], 0)
   input_image3 = cv2.imread(sys.argv[4], 0) 

   function_launch = {
   1 : Perspective_warping,
   2 : Cylindrical_warping,
   3 : Bonus_perspective_warping,
   4 : Bonus_cylindrical_warping}

   # Call the function
   function_launch[question_number](input_image1, input_image2, input_image3)
