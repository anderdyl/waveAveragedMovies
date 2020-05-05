import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import cv2
import h5py
import scipy.io
from scipy.spatial import distance
import pickle
from netCDF4 import Dataset
import xarray as xr
from ocmIO import radonCurrent
from ocmIO import vBar



def draw_flow(img, flow, step=16):
    global arrows
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    posLines = np.where((fy > 0))
    negLines = np.where((fy < 0))
    maxFy = np.max(np.abs(fy))
    for i in range(len(posLines[0])):
        tempFy = int(np.ceil((fy[posLines[0][i]]/maxFy)*255))
        cv.polylines(vis, np.int32([lines[posLines[0][i]]]), 0, (255, 0, 0))
    cv.polylines(vis, lines[negLines], 0, (0, 0, 255))

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

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})


scaleNum = 40
headwidthNum = 2
shaftWidth = 0.0045
dstep = 12
outputName = 'opticalFlowDuck1572548400_12s_2dt.avi'
rawName = 'rawOpticalFlowDuck1572548400_12s_2dt.avi'
wamAlone = 'duckWAM1572548400_12s_2dt.avi'
data = Dataset('wam1572548400_12s_2dt.nc')
file = 'wam1572548400_12s_2dt.nc'
wamPickleFlow = 'oct22wam1572548400FlowFields_unfiltered.pickle'

DT = 2
dx = 1
wamFrames_subset = data['merged']
nn = wamFrames_subset.shape[0]
nm = wamFrames_subset.shape[1]
mp = wamFrames_subset.shape[2]

frame_width = nm
frame_height = nn
#out2 = cv.VideoWriter('oystervilleWAM.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out2 = cv.VideoWriter(wamAlone,cv.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height),0)

for i in range(mp):
    finalImg = wamFrames_subset[:,:,i].astype(np.uint8)

    out2.write(np.flipud(finalImg))
out2.release()
cv.destroyAllWindows()



from scipy.stats.kde import gaussian_kde
from numpy import linspace
import matplotlib.patches as patches
# create fake data
data = wamFrames_subset[:,:,0].flatten().filled()
data0 = wamFrames_subset[625:650,100:125,0].flatten().filled()
data50 = wamFrames_subset[750:775,100:125,0].flatten().filled()
data100 = wamFrames_subset[825:850,140:165,0].flatten().filled()
data150 = wamFrames_subset[1015:1030,105:130,0].flatten().filled()
data200 = wamFrames_subset[900:925,145:170,0].flatten().filled()


# this create the kernel, given an array it will estimate the probability over that values
kde = gaussian_kde(data)
kde0 = gaussian_kde(data0)
kde50 = gaussian_kde(data50)
kde100 = gaussian_kde(data100)
kde150 = gaussian_kde(data150)
kde200 = gaussian_kde(data200)


# these are the values over wich your kernel will be evaluated
dist_space = linspace(0, 255, 255)
# plot the results
fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot2grid((4,5), (0,0), rowspan=4, colspan=3)
ax1.plot(dist_space, kde(dist_space),'k', linewidth=2)
ax1.plot( dist_space, kde0(dist_space),'red' )
ax1.plot( dist_space, kde50(dist_space),'orange' )
ax1.plot( dist_space, kde100(dist_space),'blue' )
ax1.plot( dist_space, kde150(dist_space),'green')
ax1.plot( dist_space, kde200(dist_space),'magenta')


ax2 = plt.subplot2grid((4,5), (0,3), rowspan=4, colspan=2)
ax2.imshow(np.flipud(wamFrames_subset[600:1600,0:350,0].astype(np.uint8)),cmap='gray',vmin=0,vmax=255)
rect = patches.Rectangle((100,1600-650),25,25,linewidth=1,edgecolor='red',facecolor='none')
ax2.add_patch(rect)
rect2 = patches.Rectangle((100,1600-775),25,25,linewidth=1,edgecolor='orange',facecolor='none')
ax2.add_patch(rect2)
rect3 = patches.Rectangle((140,1600-850),25,25,linewidth=1,edgecolor='blue',facecolor='none')
ax2.add_patch(rect3)
rect4 = patches.Rectangle((105,1600-1030),25,25,linewidth=1,edgecolor='green',facecolor='none')
ax2.add_patch(rect4)
rect5 = patches.Rectangle((145,1600-925),25,25,linewidth=1,edgecolor='magenta',facecolor='none')
ax2.add_patch(rect5)



sizeGradient = 8


file = '/media/dylananderson/LaCie/netcdfs/merged1572548400.nc'

data = xr.open_dataset(file)
merged = data['merged'].values
x = data['xFRF'].values
y = data['yFRF'].values
imTime = data['t'].values




# current bin is 815 to 839
# S0 y = 827, x = 172 coverts to...

# lets go for y = 700, x = 180
targetx = 200
targety = 800

# The vBar data set is
#       500:0.5:1499.5
#       80:0.5:349..5
ocmX = targetx*2-80
ocmYmin = (targety-12)*2-500
ocmYmax = (targety+12)*2-500

# The WAM data set is... so these guys have the north part of the domain close to the origin... options are to
# calculate a different
#       -200:1:1400
#       60:1:410
wamY = (1600-(targety+200))
wamX = targetx-60



timeloops = np.arange(0, 120*16, 5)
ocmVBAR = np.nan * np.ones((len(timeloops)))
vBarCispan = np.nan * np.ones((len(timeloops)))
vBarChi2 = np.nan * np.ones((len(timeloops)))
vBarProb = np.nan * np.ones((len(timeloops)))
ocmTIME = np.nan * np.ones((len(timeloops)))

for tc in range(len(timeloops)):
    imYind = np.where((y>=(targety-int(sizeGradient/2))) & (y<=(targety+int(sizeGradient/2))))#np.where((y >= yloops[yc]) & (y <= (yloops[yc] + 30)))
    imXind = np.where((x==targetx))#np.where((x == xloops[xc]))
    # imTind = np.arange(timeloops[tc],(timeloops[tc]+(120*2-1)),1)

    stack = np.fliplr(merged[imYind[0], imXind[0], timeloops[tc]:(timeloops[tc] + (64 * 1 + 1))].T)

    bad = np.argwhere(np.isnan(stack))
    stack[bad] = np.nanmean(stack)
    ogStack = stack.copy()

    ystack = y[imYind]
    xstack = x[imXind] * np.ones((np.shape(ystack)))

    imTimeSubset = np.arange(timeloops[tc], (timeloops[tc] + (64 * 1 + 1)), 1)
    imSeconds = np.arange(0, len(imTimeSubset), 0.5)

    #radonResult = radonCurrent(stack, imSeconds, ystack[0:-1], xstack[0:-1])

    vBarResult = vBar(stack, ystack, xstack)
    ocmVBAR[tc] = vBarResult['meanV']
    vBarCispan[tc] = vBarResult['cispan']
    vBarChi2[tc] = vBarResult['chi2']
    vBarProb[tc] = vBarResult['prob']
    ocmTIME[tc] = np.mean(imTimeSubset)/2

#
#
cap = cv.VideoCapture(wamAlone)
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print( length )



#files = files[1:180]
#files_path = files_path[1:180]
allFlow = []
allFlowY = []
allFlowX = []

for i in range(length):

    ret, frame = cap.read()

    img = frame
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(img)
    # Sets image saturation to maximum
    mask[..., 1] = 255

    if i == 0:
        prev_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        nanpoints = np.where((np.isnan(prev_gray)))
        prev_gray[nanpoints] = 0
        arrows = []

        scale_percent = 120  # percent of original size
        width = int(prev_gray.shape[1] * scale_percent / 100)
        height = int(prev_gray.shape[0] * scale_percent / 100)
        dim = (width, height)

        nn, nm = np.shape(prev_gray)
        frame_width = nm
        frame_height = nn
        out2 = cv.VideoWriter(rawName,cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,(width, height)) #,cv.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height),0)


    else:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        nanpoints = np.where((np.isnan(gray)))
        gray[nanpoints] = 0
        if i == 1:
            flagged = cv.OPTFLOW_FARNEBACK_GAUSSIAN
            flow = cv.calcOpticalFlowFarneback(prev=prev_gray.astype(np.uint8), next=gray.astype(np.uint8), flow=None, pyr_scale=0.5, levels=3, winsize=6, iterations=3, poly_n=7, poly_sigma=1.5, flags=flagged)
        else:
            #flagged += cv.OPTFLOW_USE_INITIAL_FLOW
            #flow = cv.calcOpticalFlowFarneback(prev=prev_gray.astype(np.uint8), next=gray.astype(np.uint8), flow=newFlow, pyr_scale=0.5, levels=3, winsize=12, iterations=3, poly_n=7, poly_sigma=1.5, flags=flagged)
            flow = cv.calcOpticalFlowFarneback(prev=prev_gray.astype(np.uint8), next=gray.astype(np.uint8), flow=None, pyr_scale=0.5, levels=3, winsize=6, iterations=3, poly_n=7, poly_sigma=1.5, flags=flagged)

        flow = flow
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
        # finalImg = draw_flow(frame_b.astype(np.uint8), flow, step=16)
        finalImg = draw_flow(gray.astype(np.uint8), flow, step=8)

        allFlow.append(flow)
        allFlowX.append(flow[:,:,0])
        allFlowY.append(flow[:,:,1])
        # print(arrows)
        # cv.imshow("input", gray)
        # newFlow = np.mean(allFlow,axis=0)

        scale_percent = 120  # percent of original size
        width = int(finalImg.shape[1] * scale_percent / 100)
        height = int(finalImg.shape[0] * scale_percent / 100)
        dim = (width, height)
        # #resize image
        resized = cv.resize(finalImg, dim, interpolation=cv.INTER_AREA)

        # cv.imshow('gradients',np.hstack([gray,gx]))
        # cv.imshow('gradients',np.hstack([gx,gy,mag]))
        # cv.imshow('flow', finalImg)
        cv.imshow('flow', resized)


        prev_gray = gray

        out2.write(resized)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


out2.release()

cv.destroyAllWindows()

timexX = np.mean(allFlowX, axis=0)/DT
timexY = np.mean(allFlowY, axis=0)/DT

#magnitudeT, angleT = cv.cartToPolar(timexX, timexY)
#flow2 = np.dstack((timexX,timexY))

if dx == 1:
    xi = np.arange(60,410)
    #ys = np.arange(400,1400)
    yi = np.arange(-200,1400)
else:
    xi = np.arange(60,410,0.5)
    #ys = np.arange(400,1400)
    yi = np.arange(-200,1400,0.5)
OGxgrid, OGygrid = np.meshgrid(xi,yi)




def ExtractVelocity(lst):
    #return [item[1600-1028,118] for item in lst]
    return [item[wamY,wamX] for item in lst]

veloc = np.divide(ExtractVelocity(allFlowY),DT)

def Extract(lst):
    return [item[wamY,wamX] for item in lst]

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

IQR = np.nan * np.ones((mp,))

mediandiff = np.nan * np.ones((mp,))
stddiff = np.nan * np.ones((mp,))
medianDO = np.nan * np.ones((mp,))
stdDO = np.nan * np.ones((mp,))
medianD = np.nan * np.ones((mp,))
stdD = np.nan * np.ones((mp,))
justf = np.nan * np.ones((mp,))
kl = np.nan * np.ones((mp,))

timeD = np.arange(0,mp)*DT/60

for tt in range(mp-1):

    fig = plt.figure(figsize=(12,10))
    ax2 = plt.subplot2grid((6,6), (0,4), rowspan=6, colspan=2)

    ax2.imshow(np.flipud(wamFrames_subset[0:1600,0:350,tt+1].astype(np.uint8)),cmap='gray',vmin=0,vmax=255)

    ax1 = plt.subplot2grid((6,6), (0,0), rowspan=2, colspan=4)

    #data200 = wamFrames_subset[900:925,145:170,tt].flatten().filled()
    data2001 = wamFrames_subset[wamY-int(sizeGradient/2):wamY+int(sizeGradient/2), wamX-int(sizeGradient/2):wamX+int(sizeGradient/2), tt+1].flatten().filled()
    kde2001 = gaussian_kde(data2001)
    ax1.plot( dist_space, kde2001(dist_space),'magenta')
    medianD[tt] = np.nanmean(data2001)
    stdD[tt] = np.nanstd(data2001)

    data2000 = wamFrames_subset[wamY-int(sizeGradient/2):wamY+int(sizeGradient/2), wamX-int(sizeGradient/2):wamX+int(sizeGradient/2), tt].flatten().filled()
    kde2000 = gaussian_kde(data2000)
    #ax1.plot( dist_space, kde2000(dist_space),'magenta')
    mediandiff[tt] = np.nanmean(data2001) - np.nanmean(data2000)
    stddiff[tt] = np.nanstd(data2001) - np.nanstd(data2000)

    dist1 = kde2001(dist_space)
    zeroind1 = np.where((dist1==0))
    dist1[zeroind1] = 0.000000000001
    dist2 = kde2000(dist_space)
    zeroind2 = np.where((dist2==0))
    dist2[zeroind2] = 0.000000000001
    kl[tt] = kl_divergence(dist1, dist2)

    subset = wamFrames_subset[wamY-int(sizeGradient/2):wamY+int(sizeGradient/2), wamX-int(sizeGradient/2):wamX+int(sizeGradient/2), tt+1].flatten().filled()
    temp = np.where((subset > 100))
    foamfractionT = len(temp[0])/np.square(sizeGradient)
    IQR[tt] = scipy.stats.iqr(subset, rng=(5, 95),)
    justf[tt] = foamfractionT

    ax3 = plt.subplot2grid((6,6), (2,0), rowspan=1, colspan=4)
    #ax3.plot(timeD, mediandiff)
    ax3.plot(timeD,justf)
    ax3.set_xlim([0, 17])
    #ax3.set_ylim([-10, 10])
    ax3.set_ylim([-0.02, 1.02])
    #ax3.set_ylabel('d(mean)/dt')
    ax3.set_ylabel('foam fraction (>110)')
    ax4 = plt.subplot2grid((6,6), (3,0), rowspan=1, colspan=4)
    #ax4.plot(timeD, stdD)
    ax4.plot(timeD,kl)
    ax4.set_xlim([0,17])
    ax4.set_ylim([0,4])
    #ax4.set_xlabel('time (mins)')
    ax4.set_ylabel('kl(box)')

    ax5 = plt.subplot2grid((6,6), (4,0), rowspan=2, colspan=4)



    #ax5.plot(timeD[0:tt], IQR[0:tt])
    #ax5.plot(ocmTIME/60,ocmVBAR,'.-',label='vBar')
    ax5.scatter(ocmTIME/60,ocmVBAR,10,vBarProb,label='vBar')
    ax5.plot(timeD[0:tt], veloc[0:tt],'--',label='WAM')
    ax5.set_xlim([0,17])
    ax5.set_ylim([-1.5, 1.5])
    #ax5.set_xlabel('time (mins)')
    ax5.set_ylabel('vBar')

    # ax6 = plt.subplot2grid((6,6), (5,0), rowspan=1, colspan=4)
    # ax6.plot(timeD[0:tt], veloc[0:tt])
    # ax6.set_ylim([-0.75, 0.75])
    # ax6.set_xlim([0,17])
    # ax6.set_xlabel('time (mins)')
    # ax6.set_ylabel('velocity')


    if tt > 0:
        data200 = wamFrames_subset[wamY-int(sizeGradient/2):wamY+int(sizeGradient/2), wamX-int(sizeGradient/2):wamX+int(sizeGradient/2), tt-1].flatten().filled()
        kde200 = gaussian_kde(data200)
        ax1.plot(dist_space, kde200(dist_space), color=[0.8, 0.8, 0.8])

    if tt > 1:
        data200 = wamFrames_subset[wamY-int(sizeGradient/2):wamY+int(sizeGradient/2), wamX-int(sizeGradient/2):wamX+int(sizeGradient/2), tt-2].flatten().filled()
        kde200 = gaussian_kde(data200)
        ax1.plot(dist_space, kde200(dist_space), color=[0.5, 0.5, 0.5])

    if tt > 2:
        data200 = wamFrames_subset[wamY-int(sizeGradient/2):wamY+int(sizeGradient/2), wamX-int(sizeGradient/2):wamX+int(sizeGradient/2), tt-3].flatten().filled()
        kde200 = gaussian_kde(data200)
        ax1.plot(dist_space, kde200(dist_space), color=[0.2, 0.2, 0.2])

    ax1.set_ylim([0, 0.1])
    ax1.set_xlabel('Brightness')
    ax1.set_ylabel('Probability')


    rect5 = patches.Rectangle((wamX-int(sizeGradient/2),wamY-int(sizeGradient/2)),sizeGradient,sizeGradient,linewidth=1,edgecolor='magenta',facecolor='none')
    ax2.add_patch(rect5)

    if tt < 10:
        plt.savefig('/home/dylananderson/projects/drifters/distributionMovie1/frame000'+str(tt)+'.png')
    elif tt < 100:
        plt.savefig('/home/dylananderson/projects/drifters/distributionMovie1/frame00'+str(tt)+'.png')
    elif tt < 1000:
        plt.savefig('/home/dylananderson/projects/drifters/distributionMovie1/frame0'+str(tt)+'.png')
    else:
        plt.savefig('/home/dylananderson/projects/drifters/distributionMovie1/frame' + str(tt) + '.png')

    plt.close()



nanind = np.where((kl>10))
kl[nanind] = np.nan

fig = plt.figure(figsize=(10,10))
# plt.scatter(veloc,kl[0:-1],10,mediandiff[0:-1],vmin=-5,vmax=5,cmap='RdBu_r')
plt.scatter(veloc,mediandiff[0:-1],10,kl[0:-1],vmin=0,vmax=1,cmap='RdBu_r')
#plt.scatter(mediandiff[0:-1],kl[0:-1],10,veloc,vmin=-0.5,vmax=0.5,cmap='RdBu_r')

plt.colorbar()

fig = plt.figure(figsize=(10,10))
kdevel = gaussian_kde(veloc)
velspace = np.arange(-1,1,0.01)
plt.plot(velspace, kdevel(velspace), 'magenta')

geomorphdir = '/home/dylananderson/projects/drifters/distributionMovie1/'

files = os.listdir(geomorphdir)
files.sort()
files_path = [os.path.join(geomorphdir,x) for x in os.listdir(geomorphdir)]
files_path.sort()
files_path=files_path[1:len(allFlow)]
import cv2

frame = cv2.imread(files_path[0])
height, width, layers = frame.shape
forcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('distributionsNeighborhoodOct22.avi', forcc, 16, (width, height))
for image in files_path:
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()

