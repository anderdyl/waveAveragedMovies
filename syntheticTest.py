


import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv



def draw_flow(img, flow, step=16):
    global arrows
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    #lines = np.int32(lines + 0.5)
    lines = np.int32(lines + 0.5)

    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    posLines = np.where((fy > 0))
    negLines = np.where((fy < 0))

    maxFy = np.max(np.abs(fy))

    #cv.arrowedLine(vis, lines[posLines], 0, (255, 0, 0))
    #cv.arrowedLine(vis, lines[negLines], 0, (0, 0, 255))
    for i in range(len(posLines[0])):
        tempFy = int(np.ceil((fy[posLines[0][i]]/maxFy)*255))
        #print(tempFy)
        cv.polylines(vis, np.int32([lines[posLines[0][i]]]), 0, (255, 0, 0))

    cv.polylines(vis, lines[negLines], 0, (0, 0, 255))

    #for (x1, y1), (x2, y2) in lines:
    #    arrows.append([x1,y1, 100*math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))])
    #    cv.circle(vis, (x1, y1), 1, (255, 255, 255), -1)
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


r = 20
r13 = int(np.round(r*np.sqrt(3)/2))
r12 = int(r/2)
r22 = int(np.round(r*np.sqrt(2)/2))
r10 = int(r*np.cos(15*np.pi/180))
r01 = int(r*np.sin(15*np.pi/180))
r25 = int(r*np.cos(52*np.pi/180))
r52 = int(r*np.sin(52*np.pi/180))
r38 = int(r*np.cos(38*np.pi/180))
r83 = int(r*np.sin(38*np.pi/180))
r5 = int(r*np.cos(5*np.pi/180))
r05 = int(r*np.sin(5*np.pi/180))

centerx = 150
centery = 40

for i in range(1420):
    im = np.zeros((300, 1500)).astype(np.uint8)

    im[centerx-r:centerx+r, centery] = 255
    im[centerx, centery-r:centery+r] = 255
    im[centerx-r12:centerx+r12, centery-r13:centery+r13] = 255
    im[centerx-r13:centerx+r13, centery-r12:centery+r12] = 255
    im[centerx-r22:centerx+r22, centery-r22:centery+r22] = 255
    im[centerx-r10:centerx+r10, centery-r01:centery+r01] = 255
    im[centerx-r01:centerx+r01, centery-r10:centery+r10] = 255
    im[centerx-r25:centerx+r25, centery-r52:centery+r52] = 255
    im[centerx-r52:centerx+r52, centery-r25:centery+r25] = 255
    im[centerx-r38:centerx+r38, centery-r83:centery+r83] = 255
    im[centerx-r83:centerx+r83, centery-r38:centery+r38] = 255
    im[centerx-r05:centerx+r05, centery-r5:centery+r5] = 255
    im[centerx-r5:centerx+r5, centery-r05:centery+r05] = 255
    centery += 1


    print('saving {}'.format(i))

    if i < 10:
        cv.imwrite('/home/dylananderson/projects/drifters/synthetic/frame000'+str(i)+'.png',im)
    elif i < 100:
        cv.imwrite('/home/dylananderson/projects/drifters/synthetic/frame00'+str(i)+'.png',im)
    elif i < 1000:
        cv.imwrite('/home/dylananderson/projects/drifters/synthetic/frame0'+str(i)+'.png',im)
    else:
        cv.imwrite('/home/dylananderson/projects/drifters/synthetic/frame'+str(i)+'.png',im)

    cv.destroyAllWindows()



geomorphdir = '/home/dylananderson/projects/drifters/synthetic/'
import os
outputName = 'allFrames.avi'
files = os.listdir(geomorphdir)

files.sort()

files_path = [os.path.join(geomorphdir,x) for x in os.listdir(geomorphdir)]

files_path.sort()
import cv2

frame = cv2.imread(files_path[0])
height, width, layers = frame.shape
forcc = cv2.VideoWriter_fourcc(*'XVID')

# full temporal resolution
video = cv2.VideoWriter(outputName, forcc, 8, (width, height))
for image in files_path:
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()

allImages = []
for image in files_path:
    allImages.append(cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2GRAY))



nsteps = np.arange(0,1420-40,20)

fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
outputName = 'allFrames_20s_10dt.avi'
video = cv2.VideoWriter(outputName, fourcc, 8, (width, height))


for i in range(len(nsteps)):
    skip = np.int(nsteps[i])
    tempFrame = np.mean(allImages[skip:skip+40], axis=0).astype(np.uint8)
    colorized = cv2.cvtColor(tempFrame,cv2.COLOR_GRAY2BGR)
    video.write(colorized)
cv2.destroyAllWindows()
video.release()


wamAlone = outputName
rawName = 'opticalFlowAllFrames_20s_10dt.avi'
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
        flagged = cv.OPTFLOW_FARNEBACK_GAUSSIAN

        if i == 1:
            flow = cv.calcOpticalFlowFarneback(prev=prev_gray.astype(np.uint8), next=gray.astype(np.uint8), flow=None, pyr_scale=0.5, levels=3, winsize=10, iterations=3, poly_n=7, poly_sigma=1.5, flags=flagged)
        else:
            flagged += cv.OPTFLOW_USE_INITIAL_FLOW
            flow = cv.calcOpticalFlowFarneback(prev=prev_gray.astype(np.uint8), next=gray.astype(np.uint8), flow=newFlow, pyr_scale=0.5, levels=3, winsize=10, iterations=3, poly_n=7, poly_sigma=1.5, flags=flagged)

        flow = flow/10
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # Opens a new window and displays the output frame
        # finalImg = draw_flow(frame_b.astype(np.uint8), flow, step=16)
        finalImg = draw_flow(gray.astype(np.uint8), flow*10, step=8)

        allFlow.append(flow)
        allFlowX.append(flow[:,:,0])
        allFlowY.append(flow[:,:,1])
        # print(arrows)
        # cv.imshow("input", gray)
        newFlow = np.mean(allFlow,axis=0)

        scale_percent = 120  # percent of original size
        width = int(finalImg.shape[1] * scale_percent / 100)
        height = int(finalImg.shape[0] * scale_percent / 100)
        dim = (width, height)
        # #resize image
        resized = cv.resize(finalImg, dim, interpolation=cv.INTER_AREA)
        cv.imshow('flow', resized)


        prev_gray = gray

        out2.write(resized)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


out2.release()

cv.destroyAllWindows()


DT = 10
dx = 1


import copy

nanFlowX = copy.deepcopy(allFlowX)
nanFlowY = copy.deepcopy(allFlowY)
OGxgrid, OGygrid = np.meshgrid(np.arange(0,300,1),np.arange(0,1500,1))

for iii in range(67):
    #inputBrightness = np.ma.filled(np.flipud(wamFrames_subset[:,:,iii].astype(np.uint8)),0)
    #inputBrightness2 = np.ma.filled(np.flipud(wamFrames_subset[:,:,iii+1].astype(np.uint8)),0)
    #diffBrightness = np.abs(inputBrightness2.astype(np.float64)-inputBrightness.astype(np.float64))
    #lap64 = cv.Laplacian(diffBrightness,cv.CV_64F)
    #abs_lap64f = np.absolute(lap64)
    #lap_8u = np.uint8(abs_lap64f)
    flowX = nanFlowX[iii].copy()
    flowY = nanFlowY[iii].copy()
    hypot = np.hypot(flowX, flowY)
    magnitudeT, angleT = cv.cartToPolar(flowX, flowY)

    tooLow = np.where(hypot < 0.05)
    nanFlowX[iii][tooLow] = np.nan
    nanFlowY[iii][tooLow] = np.nan
    #plt.plot(diffBrightness,hypot,'.')





fig = plt.figure(figsize=(8,10))
ax1 = plt.subplot2grid((1,2),(0,0),rowspan=1,colspan=1)
#ax1.imshow(gray.T/1.5, cmap=plt.cm.gray, vmin=0, vmax=255, extent = [0, 300, 0, 1500])

import matplotlib.colors

fxGlobal = np.nanmean(allFlowX,axis=0)
fyGlobal = np.nanmean(allFlowY,axis=0)

Mgridhypot = np.hypot(fxGlobal,fyGlobal)
Mgrid = fxGlobal
minfy = -2#0.17#np.max(np.abs(M))
maxfy = 2#0.17#np.max(np.abs(M))
norm = matplotlib.colors.Normalize(vmin=minfy, vmax=maxfy)


# zeroV = np.where(Mgridhypot < 0.001)
# fxgrid[zeroV] = np.nan
# fygrid[zeroV] = np.nan

#lw = 0.5 * Mgridhypot / 1.5 #Mgrid.max()
#sc = ax1.streamplot(OGxgrid, OGygrid, fyGlobal.T, -fxGlobal.T, density=3, color=Mgrid.T, cmap='RdBu', norm=norm)#, linewidth=lw.T)
sc = ax1.pcolor(OGxgrid,OGygrid,fxGlobal.T)
    #fig.colorbar(sc.lines)
#plt.ylim([450, 1400])
#plt.xlim([80, 350])
#fig.colorbar(sc.lines)
fig.colorbar(sc)

fxGlobalNAN = np.nanmax(nanFlowX,axis=0)
fyGlobalNAN = np.nanmax(nanFlowY,axis=0)

MgridhypotNAN = np.hypot(fxGlobalNAN,fyGlobalNAN)
MgridNAN = fxGlobalNAN
minfy = -2#0.17#np.max(np.abs(M))
maxfy = 2#0.17#np.max(np.abs(M))
norm = matplotlib.colors.Normalize(vmin=minfy, vmax=maxfy)

lw = 0.5 * MgridhypotNAN / 1.5 #Mgrid.max()
ax2 = plt.subplot2grid((1,2),(0,1),rowspan=1,colspan=1)
#ax2.imshow(gray.T/1.5, cmap=plt.cm.gray, vmin=0, vmax=255, extent = [0, 300, 0, 1500])

#sc2 = ax2.streamplot(OGxgrid, OGygrid, fyGlobal.T, -fxGlobal.T, density=3, color=Mgrid.T, cmap='RdBu', norm=norm)#, linewidth=lw.T)
sc2 = ax2.pcolor(OGxgrid,OGygrid,fxGlobalNAN.T)
fig.colorbar(sc2)
#fig.colorbar(sc2.lines)
#plt.ylim([450, 1400])
#plt.xlim([80, 350])


from operator import itemgetter


def Extract(lst):
    return [item[150,600] for item in lst]

print(Extract(allFlowX))
time = np.arange(10,1420/2-20,10)
plt.plot(time,Extract(allFlowX))

xs = np.arange(0,300)
#ys = np.arange(400,1400)
ys = np.arange(0,1500)

newygrid, newxgrid = np.meshgrid(ys,xs)

snaps = list()
data = allFlowX[0] + 1j * allFlowY[0]
data = data.flatten()
for i in range(len(allFlow)):

    complexF = allFlowX[i] + 1j * allFlowY[i]
    if i > 0:
        data = np.vstack((data,complexF.flatten()))
    #snaps.append(complexF[0:1000,:])
    snaps.append(complexF)

#
# data = data.T
# c = np.matmul(np.conj(data).T,data)/np.shape(data)[0]
#
#
# import scipy.linalg as la
# import numpy.linalg as npla
#
# lamda, loadings = la.eigh(c)
#
# lamda2, loadings2 = npla.eig(c)
#
# ind = np.argsort(lamda[::-1])
#
# lamda[::-1].sort()
#
# loadings = loadings[:,ind]
#
# pcs = np.dot(data, loadings)# / np.sqrt(lamda)
# loadings = loadings# * np.sqrt(lamda)
# pcsreal = np.real(pcs[:,0:200])
# pcsimag = np.imag(pcs[:,0:200])
# eofreal = np.real(loadings[:,0:200])
# eofimag = np.imag(loadings[:,0:200])
# S = np.power(loadings*np.conj(loadings),0.5) * np.sqrt(lamda)
#
# theta = np.arctan2(eofimag,eofreal)
# theta2 = theta*180/np.pi
#
# Rt = np.power(pcs*np.conj(pcs),0.5) / np.sqrt(lamda)
#
# phit = np.arctan2(pcsimag,pcsreal)
# phit2 = phit*180/np.pi
#
# mode = 0

# fig, ax = plt.subplots(2,2)
#
# ax[0,0].plot(xinterp, S[:,mode],'o')
# ax[0,1].plot(xinterp, theta2[:,mode],'o')
# ax[1,0].plot(time,Rt[:,mode],'o')
# ax[1,1].plot(time,phit2[:,mode],'o')
















#
# from pydmd import DMD
# from pydmd import MrDMD
# dmd = DMD(svd_rank=2, tlsq_rank=2, exact=True, opt=True)
# dmd.fit(snaps)
# dmd.plot_modes_2D(x=newygrid[0,:], y=newxgrid[:,0], figsize=(8,8))
#
#
#
# # fig = plt.figure(figsize=(18,12))
# # for id_subplot, snapshot in enumerate(dmd.reconstructed_data.T, start=1):
# #     plt.subplot(4, 4, id_subplot)
# #     plt.pcolor(newygrid, newxgrid, snapshot.reshape(newygrid.shape).real, vmin=-2, vmax=2)
# #
#
#
#
# #newxgrid = ogxgrid #[200:700,:]
# #newygrid = ogygrid #[200:700,:]
#
#
#
# fig = plt.figure(figsize=(14,10))
# ax1 = plt.subplot2grid((3,1),(0,0),rowspan=1,colspan=1)
# ax1.imshow(gray, cmap=plt.cm.gray, vmin=0, vmax=255)#, extent=[0, 350, 0, 1500])
# #u = np.flipud(np.reshape(np.real(dmd.dynamics[0,0])*np.real(dmd.modes[:,0]),(np.shape(newxgrid))))
# #v = -np.flipud(np.reshape(np.real(dmd.dynamics[0,0])*np.imag(dmd.modes[:,0]),(np.shape(newxgrid))))
# u = np.reshape(np.real(dmd.modes[:,0]),(np.shape(newxgrid)))
# v = np.reshape(np.imag(dmd.modes[:,0]),(np.shape(newxgrid)))
# #Mnew = np.hypot(u,v)/10
# Mnew = u
# minfy = -np.max(np.abs(u))
# maxfy = np.max(np.abs(u))
# norm = matplotlib.colors.Normalize(vmin=minfy, vmax=maxfy)
#
# st = ax1.streamplot(newygrid,newxgrid, u, v, density = 4, color=Mnew, cmap = 'RdBu_r', norm = norm)
# fig.colorbar(st.lines)
# #plt.xlim([np.min(xs), np.max(xs)])
# #plt.ylim([np.min(ys), np.max(ys)])
# #plt.ylim([400, 1400])
#
# ax1.set_title('DMD #1', color='w')
#
#
#
# ax2 = plt.subplot2grid((3,1),(1,0),rowspan=1,colspan=1)
# ax2.imshow(gray, cmap=plt.cm.gray, vmin=0, vmax=255)#, extent=[0, 300, 0, 1500])
# #v2 = np.flipud(np.reshape(np.real(dmd.dynamics[1,1])*np.real(dmd.modes[:,1]),(np.shape(newxgrid))))
# #u2 = -np.flipud(np.reshape(np.real(dmd.dynamics[1,1])*np.imag(dmd.modes[:,1]),(np.shape(newxgrid))))
# u2 = np.flipud(np.reshape(np.real(dmd.modes[:,1]),(np.shape(newxgrid))))
# v2 = -np.flipud(np.reshape(np.imag(dmd.modes[:,1]),(np.shape(newxgrid))))
# #Mnew2 = np.hypot(u2,v2)/15
# Mnew2 = u2
#
# minfy = -np.max(np.abs(u2))
# maxfy = np.max(np.abs(u2))
# norm2 = matplotlib.colors.Normalize(vmin=minfy, vmax=maxfy)
# st2 = ax2.streamplot(newygrid,newxgrid, u2, v2, density = 4, color=Mnew2, cmap = 'RdBu_r', norm = norm2)
# #fig.colorbar(st2.lines)
# #plt.xlim([np.min(xs), np.max(xs)])
# #plt.ylim([np.min(ys), np.max(ys)])
# #plt.ylim([400, 1400])
# ax2.set_title('DMD #2', color='w')
#
# #
# #
# # ax3 = plt.subplot2grid((1,3),(0,2),rowspan=1,colspan=1)
# # ax3.imshow(gray, cmap=plt.cm.gray, vmin=0, vmax=255)#, extent=[0, 300, 0, 1500])
# # v3 = np.flipud(np.reshape(np.real(dmd.dynamics[2,0])*np.real(dmd.modes[:,2]),(np.shape(newxgrid))))
# # u3 = -np.flipud(np.reshape(np.real(dmd.dynamics[2,0])*np.imag(dmd.modes[:,2]),(np.shape(newxgrid))))
# # #u3 = np.flipud(np.reshape(np.real(dmd.modes[:,2]),(np.shape(newxgrid))))
# # #v3 = -np.flipud(np.reshape(np.imag(dmd.modes[:,2]),(np.shape(newxgrid))))
# # #Mnew3 = np.hypot(u3,v3)/10
# # Mnew3 = v3
# #
# # minfy = -np.max(np.abs(v3))
# # maxfy = np.max(np.abs(v3))
# # norm3 = matplotlib.colors.Normalize(vmin=minfy, vmax=maxfy)
# # st3 = ax3.streamplot(newxgrid,newygrid, u3, v3, density = 4, color=Mnew3, cmap = 'RdBu_r', norm = norm3)
# # #fig.colorbar(st3.lines)
# # #plt.xlim([np.min(xs), np.max(xs)])
# # #plt.ylim([np.min(ys), np.max(ys)])
# # ax3.set_title('DMD #3', color='w')
# #
# # plt.tight_layout()
#
#



