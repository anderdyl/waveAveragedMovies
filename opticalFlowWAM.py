import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt


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

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res



# The video feed is read in as a VideoCapture object
# cap = cv.VideoCapture("shibuya.mp4")
cap = cv.VideoCapture("wam30sAverage_0pt5DT_testThresh.avi")
#cap = cv.VideoCapture("wam30sAverage_0pt5DT_test.avi")

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
first_frame = first_frame[400:,:]
from openpiv import tools, process, validation, filters, scaling, pyprocess

# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
im = np.float32(prev_gray) / 255.0

# Calculate gradient
prev_gx = cv.Sobel(im, cv.CV_32F, 1, 0, ksize=3)
prev_gy = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=3)
prev_mag, prev_angle = cv.cartToPolar(prev_gx, prev_gy, angleInDegrees=True)
frame_a = (prev_mag - np.min(prev_mag)) / (np.max(prev_mag) - np.min(prev_mag)) * 255
#frame_a = (prev_gx - np.min(prev_gx)) /(np.max(prev_gx)-np.min(prev_gx))*255
#frame_a = cv.GaussianBlur(frame_a, (3,3), 0)

arrows = []

nn, nm = np.shape(prev_gray)

frame_width = nm
frame_height = nn
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
#out2 = cv.VideoWriter('wam30sAverage_5sDT_OpticalFlow.avi',cv.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height),0)
out2 = cv.VideoWriter('wam30sAverage_5sDT_OpticalFlow.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


#prev_gray = cv.blur(prev_gray, (10,10))

# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret1, frame1 = cap.read() # skipping to 1 second
    ret2, frame2 = cap.read() # skipping to 1.5 seconds
    ret3, frame3 = cap.read() # skipping to 2 seconds
    ret4, frame4 = cap.read() # skipping to 2.5 seconds
    ret5, frame5 = cap.read() # skipping to 3 seconds
    ret6, frame6 = cap.read() # skipping to 3.5 seconds
    ret7, frame7 = cap.read() # skipping to 4 seconds
    ret8, frame8 = cap.read()  # skipping to 4.5 seconds
    ret9, frame9 = cap.read()  # skipping to 5 seconds
    #ret10, frame10 = cap.read()  # skipping to 5.5 seconds
    #ret11, frame11 = cap.read()  # skipping to 6 seconds
    #ret12, frame12 = cap.read()  # skipping to 6.5 seconds
    # ret13, frame13 = cap.read()  # skipping to 7 seconds
    # ret14, frame14 = cap.read()  # skipping to 7.5 seconds
    # ret15, frame15 = cap.read()  # skipping to 8 seconds
    # ret16, frame16 = cap.read()  # skipping to 8.5 seconds
    # ret17, frame17 = cap.read()  # skipping to 9 seconds
    # ret18, frame18 = cap.read()  # skipping to 9.5 seconds
    # ret19, frame19 = cap.read()  # skipping to 10 seconds

    ret, frame = cap.read()

    frame = frame[400:, :]
    # Opens a new window and displays the input frame
    #frameS = cv.resize(frame, (1024, 1224))
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.blur(gray, (10, 10))
    im = np.float32(gray) / 255.0

    # Calculate gradient
    gx = cv.Sobel(im, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=3)
    mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)
    frame_b = (mag - np.min(mag)) / (np.max(mag) - np.min(mag)) * 255
    #frame_b = (gx - np.min(gx)) /(np.max(gx)-np.min(gx))*255

    #frame_b = cv.GaussianBlur(frame_b, (3, 3), 0)

    # winsize = 16  # pixels
    # searchsize = 32  # pixels, search in image B
    # overlap = 8  # pixels
    # dt = 0.02  # sec
    # u0, v0, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32),
    #                                                      window_size=winsize, overlap=overlap, dt=dt,
    #                                                      search_area_size=searchsize, sig2noise_method='peak2peak')
    # # u0, v0, sig2noise = pyprocess.piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
    #
    # x, y = process.get_coordinates(image_size=frame_a.shape, window_size=winsize, overlap=overlap)
    # u1, v1, mask = validation.sig2noise_val(u0, v0, sig2noise, threshold=1.3)
    #
    # u2, v2 = filters.replace_outliers(u1, v1, method='localmean', max_iter=10, kernel_size=2)
    # x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor=96.52)
    # # x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 1 )
    #
    # #tools.save(x, y, u3, v3, mask, 'exp1_001.txt')
    # #tools.display_vector_field('exp1_001.txt', scale=100, width=0.0025)
    #
    # cv.imshow('test',frame_b.astype(np.uint8))
    #
    # plt.quiver(x,y,u3,v3)




    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
#    flow = cv.calcOpticalFlowFarneback(prev=prev_gray, next=gray, flow=None, pyr_scale=0.5, levels=3, winsize=25, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

    # LOOKS LIKE WE ARE GETTING SOMEWHERE WITH THE NORMALIZED VERSION
    flow = cv.calcOpticalFlowFarneback(prev=prev_gray.astype(np.uint8), next=gray.astype(np.uint), flow=None, pyr_scale=0.5, levels=3, winsize=24, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

    #flow = cv.calcOpticalFlowFarneback(prev=frame_a.astype(np.uint8), next=frame_b.astype(np.uint), flow=None, pyr_scale=0.5, levels=3, winsize=24, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

    flow = flow*10
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    #rgbS = cv.resize(rgb, (1024, 1224))
    arrows.clear()
    #finalImg = draw_flow(frame_b.astype(np.uint8), flow, step=16)
    finalImg = draw_flow(gray.astype(np.uint8), flow, step=16)



    print(arrows)
    #cv.imshow("input", gray)

    scale_percent = 70  # percent of original size
    width = int(finalImg.shape[1] * scale_percent / 100)
    height = int(finalImg.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(finalImg, dim, interpolation=cv.INTER_AREA)

    #cv.imshow('gradients',np.hstack([gray,gx]))
    #cv.imshow('gradients',np.hstack([gx,gy,mag]))
    #cv.imshow('flow', finalImg)
    cv.imshow('flow', resized)

    #cv.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray
    prev_gx = gx
    prev_mag = mag
    frame_a = frame_b

    out2.write(finalImg)


# Closes all the frames
#
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
#


# The following frees up resources and closes all windows
cap.release()
out2.release()

cv.destroyAllWindows()





#
# import cv2 as cv
# import numpy as np
#
# # Parameters for Shi-Tomasi corner detection
# feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# # Parameters for Lucas-Kanade optical flow
# lk_params = dict(winSize = (11,11), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# # The video feed is read in as a VideoCapture object
# #cap = cv.VideoCapture("wamImageSpaceC2_1572548400_15seconds.avi")
# cap = cv.VideoCapture("wamImageSpaceC2_1572548400_20seconds_3secondDT.avi")
#
# # Variable for color to draw optical flow track
# color = (0, 255, 0)
# # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
# ret, first_frame = cap.read()
# # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
# prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# #prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# #prev_gray = cv.blur(prev_gray, (20,20))
# # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
# prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
# # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
# mask = np.zeros_like(first_frame)
#
# while(cap.isOpened()):
#     # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
#     ret, frame = cap.read()
#     # Converts each frame to grayscale - we previously only converted the first frame to grayscale
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # gray = cv.blur(gray, (20, 20))
#     # Calculates sparse optical flow by Lucas-Kanade method
#     # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
#     next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
#     # Selects good feature points for previous position
#     good_old = prev[status == 1]
#     # Selects good feature points for next position
#     good_new = next[status == 1]
#     # Draws the optical flow tracks
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         # Returns a contiguous flattened array as (x, y) coordinates for new point
#         a, b = new.ravel()
#         # Returns a contiguous flattened array as (x, y) coordinates for old point
#         c, d = old.ravel()
#         # Draws line between new and old position with green color and 2 thickness
#         mask = cv.line(mask, (a, b), (c, d), color, 2)
#         # Draws filled circle (thickness of -1) at new position with green color and radius of 3
#         frame = cv.circle(frame, (a, b), 3, color, -1)
#     # Overlays the optical flow tracks on the original frame
#     output = cv.add(frame, mask)
#     # Updates previous frame
#     prev_gray = gray.copy()
#     # Updates previous good feature points
#     prev = good_new.reshape(-1, 1, 2)
#     # Opens a new window and displays the output frame
#     cv.imshow("sparse optical flow", output)
#     # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break
# # The following frees up resources and closes all windows
# cap.release()
# cv.destroyAllWindows()
