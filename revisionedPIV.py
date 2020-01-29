

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import math


def draw_flow(img, flow, step=16):
    global arrows
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    #lines = np.int32(lines + 0.5)
    lines = np.int32(lines + 0.5)

    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        arrows.append([x1,y1, 100*math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))])
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

from openpiv import tools, process, validation, filters, scaling, pyprocess

cap = cv.VideoCapture("wam30sAverage_0pt5DT_testThresh.avi")
#cap = cv.VideoCapture("wam30sAverage_0pt5DT_test.avi")

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
#first_frame = cv.imread('B005_1.tif')
ret, first_frame = cap.read()
first_frame = first_frame[:, 100:]
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
im = np.float32(prev_gray) / 255.0
# Calculate gradient
prev_gx = cv.Sobel(im, cv.CV_32F, 1, 0, ksize=3)
prev_gy = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=3)
prev_mag, prev_angle = cv.cartToPolar(prev_gx, prev_gy, angleInDegrees=True)
#frame_a = (prev_gx - np.min(prev_gx)) /(np.max(prev_gx)-np.min(prev_gx))*255
frame_a = (prev_mag - np.min(prev_mag)) /(np.max(prev_mag)-np.min(prev_mag))*255

# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
# ret, frame = cap.read()
#
ret, frame = cap.read()
frame = frame[:, 100:]
#frame = cv.imread('B005_2.tif')
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
im = np.float32(gray) / 255.0
gx = cv.Sobel(im, cv.CV_32F, 1, 0, ksize=3)
gy = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=3)
mag, angle1 = cv.cartToPolar(gx, gy, angleInDegrees=True)


#frame_b = (gx - np.min(gx)) /(np.max(gx)-np.min(gx))*255
frame_b = (mag - np.min(mag)) /(np.max(mag)-np.min(mag))*255




fig,ax = plt.subplots(1,2)
ax[0].imshow(frame_a,cmap=plt.cm.gray)
ax[1].imshow(frame_b,cmap=plt.cm.gray)

winsize = 16 # pixels
searchsize = 32  # pixels, search in image B
overlap = 8 # pixels
dt = 0.02 # sec

u0, v0, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
#u0, v0, sig2noise = pyprocess.piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )

x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )
u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3 )

u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=10, kernel_size=2)
x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 96.52 )
#x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 1 )

tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )
tools.display_vector_field('exp1_001.txt', scale=100, width=0.0025)


#plt.quiver(x,y,u0,v0)
#

arrows = []

mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

flow = cv.calcOpticalFlowFarneback(prev=frame_a.astype(np.uint8), next=frame_b.astype(np.uint8), flow=None, pyr_scale=0.5, levels=2, winsize=16, iterations=2,
                                   poly_n=7, poly_sigma=1.5, flags=0)

flow = flow*100
# Computes the magnitude and angle of the 2D vectors
magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
# Sets image hue according to the optical flow direction
mask[..., 0] = angle * 180 / np.pi / 2
# Sets image value according to the optical flow magnitude (normalized)
mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
# Converts HSV to RGB (BGR) color representation
rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
# Opens a new window and displays the output frame
# rgbS = cv.resize(rgb, (1024, 1224))
arrows.clear()
finalImg = draw_flow(frame_b.astype(np.uint8), flow, step=16)
print(arrows)
cv.imshow("input", frame_b.astype(np.uint8))

# cv.imshow('gradients',np.hstack([gray,gx]))
# cv.imshow('gradients',np.hstack([gx,gy,mag]))
cv.imshow('flow', finalImg)