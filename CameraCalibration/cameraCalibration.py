#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('./images/example/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

'''
#Compute mean of reprojection error
tot_error=0
total_points=0
for i in range(len(objpoints)):
    reprojected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    reprojected_points=reprojected_points.reshape(-1,2)
    tot_error+=np.sum(np.abs(imgpoints[i]-reprojected_points)**2)
    total_points+=len(objpoints[i])

mean_error=np.sqrt(tot_error/total_points)
print ("Mean reprojection error: ", mean_error)
'''
################
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "RMS(RP)E: {}".format(mean_error/len(objpoints)) )

mean_error2 = 0
for i in range(len(objpoints)):
    imgpoints3, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error2 = cv2.norm(imgpoints[i], imgpoints3, cv2.NORM_L1)/len(imgpoints3)
    mean_error2 += error2
print( "MA(RP)E: {}".format(mean_error2/len(objpoints)) )
mse = np.sqrt(mean_error2 / len(objpoints))
print("MS(RP)E:", mse)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints4, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints4, cv2.NORM_L2SQR)/len(imgpoints4)
    mean_error += error
print( "MS(RP)E 2: {}".format(mean_error/len(objpoints)) )

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
np.save('mtx.npy', mtx) # save the camera matrix
np.save('dist.npy', dist) # save the distortion coefficients

#mtx = np.load('mtx.npy') # load the camera matrix
#dist = np.load('dist.npy') # load the distortion coefficients
