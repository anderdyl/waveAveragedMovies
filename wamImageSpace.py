
import argusIO
import time as T
import cv2
import matplotlib.pyplot as plt
import numpy as np

folder = '/media/dylananderson/Elements/argus02bFullFrame/2019/'
#folder = '/mnt/gaia/peeler/argus/argus02bFullFrame/2019/'

day = '/304_Oct.31/'
#day = '/322_Nov.18/'
file = '1572548400.Thu.Oct.31_19_00_00.GMT.2019.argus02b.'
#file = '1572552000.Thu.Oct.31_20_00_00.GMT.2019.argus02b.'
#file = '1572553800.Thu.Oct.31_20_30_00.GMT.2019.argus02b.'
#file = '1573842600.Fri.Nov.15_18_30_00.GMT.2019.argus02b.'
#file = '1574089200.Mon.Nov.18_15_00_00.GMT.2019.argus02b.'
cams = ['c2']#, 'c2', 'c3', 'c4']#,'c5','c6']
paths = [(folder + cams[i] + day + file + cams[i] + '.raw') for i in range(len(cams))]


offset = 6
nFramesOffset = offset

# short window
timeAverage = 25
nFrames = timeAverage*2
#delT = nFrames - np.round(nFrames/1.08)
#nsteps = np.arange(0,120*17-nFrames,delT)

# long window
timeAverageL = timeAverage + offset
nFramesL = timeAverageL*2
delT = 1
#delT = nFrames - np.round(nFrames/1.08)
#nsteps = np.arange(0,120*4-nFramesL,delT)
nsteps = np.arange(0,1,delT)



#skip = 0
yamlLoc = '/home/dylananderson/projects/drifters/cameraData.yml'

cameras = dict()

start_time = T.time()

smoothimR = np.zeros((2048, 2448, len(nsteps)), dtype='uint8')
smoothimG = np.zeros((2048, 2448, len(nsteps)), dtype='uint8')
smoothimB = np.zeros((2048, 2448, len(nsteps)), dtype='uint8')

smoothimRL = np.zeros((2048, 2448, len(nsteps)), dtype='uint8')
smoothimGL = np.zeros((2048, 2448, len(nsteps)), dtype='uint8')
smoothimBL = np.zeros((2048, 2448, len(nsteps)), dtype='uint8')

# #merged = np.uint8((merged-np.min(merged))/(np.max(merged)-np.min(merged)))
#

for i in range(len(cams)):
    for p in range(len(nsteps)):
        skip = np.int(nsteps[p])
        cameras[cams[i]] = argusIO.cameraIO(cameraID=cams[i], rawPath=paths[i], nFrames=nFramesL, yamlPath=yamlLoc,
                                            xMin=80, xMax=350, yMin=500, yMax=1200, dx=0.25, skip=skip)
        cameras[cams[i]].getCameraData()
        cameras[cams[i]].readRaw()
        cameras[cams[i]].deBayerRawFrameOpenCVForColor()

        # Section to get two short videos offset in time...

        smoothimR[:, :, p] = np.nanmean(cameras[cams[0]].imR[:, :, nFramesOffset:], axis=2)
        smoothimG[:, :, p] = np.nanmean(cameras[cams[0]].imG[:, :, nFramesOffset:], axis=2)
        smoothimB[:, :, p] = np.nanmean(cameras[cams[0]].imB[:, :, nFramesOffset:], axis=2)

        smoothimRL[:, :, p] = np.nanmean(cameras[cams[0]].imR[:, :,:nFrames], axis=2)
        smoothimGL[:, :, p] = np.nanmean(cameras[cams[0]].imG[:, :,:nFrames], axis=2)
        smoothimBL[:, :, p] = np.nanmean(cameras[cams[0]].imB[:, :,:nFrames], axis=2)


        frame = cv2.merge(((smoothimR[:, :, i]), (smoothimG[:, :, i]), (smoothimB[:, :, i])))
        frameL = cv2.merge(((smoothimRL[:, :, i]), (smoothimGL[:, :, i]), (smoothimBL[:, :, i])))

        imGrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imGrayscaleL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        diff32 = np.subtract(imGrayscale.astype('float32'), imGrayscaleL.astype('float32'))
        medFilt = cv2.medianBlur(diff32, 3)


        plt.figure(figsize=(10,5))
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
        ax1.imshow(imGrayscale, vmin = 0, vmax = 255)
        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
        difIM = ax2.imshow(medFilt, vmin = -np.max(medFilt), vmax = np.max(medFilt), cmap = 'RdBu')
        plt.colorbar(difIM, ax=ax2)
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
        ax3.imshow(imGrayscaleL, vmin = 0, vmax = 255)
        plt.tight_layout()
        # Section to get a shorter center video and a longer centered video
        # smoothimR[:, :, p] = np.nanmean(cameras[cams[0]].imR[:, :, timeAverageL-timeAverage:timeAverageL+timeAverage], axis=2)
        # smoothimG[:, :, p] = np.nanmean(cameras[cams[0]].imG[:, :, timeAverageL-timeAverage:timeAverageL+timeAverage], axis=2)
        # smoothimB[:, :, p] = np.nanmean(cameras[cams[0]].imB[:, :, timeAverageL-timeAverage:timeAverageL+timeAverage], axis=2)
        #
        # smoothimRL[:, :, p] = np.nanmean(cameras[cams[0]].imR, axis=2)
        # smoothimGL[:, :, p] = np.nanmean(cameras[cams[0]].imG, axis=2)
        # smoothimBL[:, :, p] = np.nanmean(cameras[cams[0]].imB, axis=2)

        if i < len(nsteps):
            del cameras
            cameras = dict()
#





#for i in range(len(cams)):
#    cameras[cams[i]] = argusIO.cameraIO(cameraID=cams[i], rawPath=paths[i], nFrames=nFrames, yamlPath=yamlLoc,
#                                        xMin=80, xMax=350, yMin=500, yMax=1200, dx=0.25, skip=skip)
#    cameras[cams[i]].getCameraData()
#    cameras[cams[i]].readRaw()
#    cameras[cams[i]].deBayerRawFrameOpenCVForColor()
    #cameras[cams[i]].deBayerRawFrame()

#imgs = cv2.merge((cameras[cams[0]].imB[:,:,0],cameras[cams[0]].imG[:,:,0],cameras[cams[0]].imR[:,:,0]))
#plt.imshow(imgs)
#plt.imshow(cameras[cams[0]].imGrayCV[:,:,0])

#
#del cameras[cams[0]].raw
#     cameras[cams[i]].uvToXY()
#     cameras[cams[i]].cropFrames()
#del cameras[cams[0]].imGrayCV
#     cameras[cams[i]].frameInterp()
#     del cameras[cams[i]].gray
#
# import numpy as np
# merged = np.zeros((np.shape(cameras[cams[0]].grayInterp)), dtype='uint8')
#
# for i in range(nFrames):
#     #combo = np.dstack((cameras[cams[0]].grayInterp[:,:,i],cameras[cams[1]].grayInterp[:,:,i],cameras[cams[2]].grayInterp[:,:,i],cameras[cams[3]].grayInterp[:,:,i],cameras[cams[4]].grayInterp[:,:,i]))
#     #combo = np.dstack((cameras[cams[0]].grayInterp[:,:,i],cameras[cams[1]].grayInterp[:,:,i]))
#     #combo = np.dstack((cameras[cams[0]].grayInterp[:,:,i],cameras[cams[1]].grayInterp[:,:,i],cameras[cams[2]].grayInterp[:,:,i]))
#     mat1 = cameras[cams[0]].grayInterp[:,:,i].astype('float')
#     mat2 = cameras[cams[1]].grayInterp[:,:,i].astype('float')
#     mat3 = cameras[cams[2]].grayInterp[:,:,i].astype('float')
#     mat4 = cameras[cams[3]].grayInterp[:,:,i].astype('float')
#     #mat5 = cameras[cams[4]].grayInterp[:,:,i].astype('float')
#     #mat6 = cameras[cams[5]].grayInterp[:,:,i].astype('float')
#
#     mat1[mat1==0] = 'nan'
#     mat2[mat2==0] = 'nan'
#     mat3[mat3==0] = 'nan'
#     mat4[mat4==0] = 'nan'
#     #mat5[mat5==0] = 'nan'
#     #mat6[mat6==0] = 'nan'
#
#     #combo = np.nanmean(np.dstack((mat1,mat2,mat3,mat4,mat5,mat6)), axis=2)
#     combo = np.nanmean(np.dstack((mat1,mat2,mat3,mat4)), axis=2)
#     #combo = np.nanmean(np.dstack((mat1,mat2)), axis=2)
#
#     del mat1
#     del mat2
#     del mat3
#     del mat4
#     merged[:,:,i] = np.uint8(combo)
#
#     #merged[:, :, i] = cameras[cams[0]].grayInterp[:,:,i]
#
#
# #del mat5
# #del mat6
#
#
#
#
# import matplotlib.pyplot as plt
# plt.imshow(merged[:, :, 0])
#
#
# temp = file.split('.')
# import datetime
#
# time = []
# base = datetime.datetime.fromtimestamp(int(temp[0]))
# for i in range(nFrames):
#     time.append(base + datetime.timedelta(seconds=0.5*i))
# #print(time)
#
# import xarray as xr
#
# x = cameras[cams[0]].xnew
# y = cameras[cams[0]].ynew
#
# encoding = {'merged':{'dtype':'uint8','_FillValue':0},
#             'xFRF':{'dtype':'float64','_FillValue':0},
#             'yFRF':{'dtype':'float64','_FillValue':0}}
#
# da_argus6 = xr.Dataset({'merged': (['yFRF', 'xFRF', 't'], merged)},
#                            coords={'yFRF': y,
#                                    'xFRF': x,
#                                    't': time})
# da_argus6.to_netcdf('merged1572553800.nc',encoding=encoding)
# #da_argus6.to_netcdf('test1572550200.nc',encoding=encoding)

# # The movie generator
#
#
# nn,nm,pp = np.shape(smoothimR)
# gray = np.ones((nn,nm,pp), dtype='uint8')
# frame_width = nm
# frame_height = nn
# #out = cv2.VideoWriter(filename='wamImageSpaceC2_1572548400_90s_5sDT_minus20.avi', fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), fps=15, frameSize=(frame_width,frame_height))
# out2 = cv2.VideoWriter('wamImageSpaceC2_1572548400_45s_offset05_05sDT.avi',cv2.VideoWriter_fourcc(*'XVID'), 15, (frame_width,frame_height), 0)
#
# for i in range(pp):
#
#
#     #frame = np.flipud(np.uint8(cuttimex[:,:,i]))
#     frame = cv2.merge(((smoothimR[:, :, i]), (smoothimG[:, :, i]), (smoothimB[:, :, i])))
#     # frame2 = cv2.merge(((smoothimR[:, :, i+1]), (smoothimG[:, :, i+1]), (smoothimB[:, :, i+1])))
#     frameL = cv2.merge(((smoothimRL[:, :, i]), (smoothimGL[:, :, i]), (smoothimBL[:, :, i])))
#
#     #frame = cv2.merge((np.uint8(cutimB[:, :, i]), np.uint8(cutimG[:, :, i]), np.uint8(cutimR[:, :, i])))
#
#     #frameS = frame-frameL
#
#     imGrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     imGrayscaleL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
#
#     ##diff = imGrayscale - imGrayscaleL
#     # imGrayscale2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#     ##gray[:,:,i] = diff
#     # diff64 = np.zeros((2048, 2448), dtype='float64')
#     #
#     diff64 = np.subtract(imGrayscale.astype('float32'),imGrayscaleL.astype('float32'))
#
#     # diff8 = (diff64-np.min(diff64)).astype(np.uint8)
#     ##negatives = np.where(diff64 < 0)
#     # #brights = np.where(diff64 > 15)
#     plus32 = (diff64-np.min(diff64))*255/np.max(diff64-np.min(diff64))
#     ##plus32[negatives] = 0
#     #plus32 = np.abs(diff64)
#
#     # #plus32[brights] = 0#np.mean(diff64) #(np.max(diff64-25))
#     #
#     ##plus32 = plus32*255/np.max(plus32)
#     # #plus32 = diff64-np.min(diff64)
#     #
#     converted = plus32.astype(np.uint8)
#     #converted = cv2.medianBlur(converted,5)
#     #converted2 = cv2.Laplacian(converted.astype('uint8'), cv2.CV_8U, ksize=13)
#     #converted2 = cv2.Sobel(converted.astype('uint8'), cv2.CV_8U, 1, 0, ksize=11)
#
#     # #smallnegative
#     # th3 = cv2.adaptiveThreshold(diff8,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,25,2)
#     # out2.write(th3)
#     out2.write(converted)
# out2.release()
#
# # Closes all the frames
# cv2.destroyAllWindows()



# time = np.arange(0,pp/2,0.5)
#
# import xarray as xr
# #
# x = np.arange(0,2448)
# y = np.arange(0,2048)
#
# #
# encoding = {'gray':{'dtype':'uint8','_FillValue':0},
#            'ximage':{'dtype':'float64','_FillValue':0},
#            'yimage':{'dtype':'float64','_FillValue':0}}
# #
# da_argus6 = xr.Dataset({'gray': (['yimage', 'ximage', 't'], gray)},
#                        coords={'yimage': y,
#                                'ximage': x,
#                                't': time})
#
# da_argus6.to_netcdf('grayWamImageSpaceC2_1572548400_20seconds_3secondDT.nc',encoding=encoding)
# # #da_argus6.to_netcdf('test1572550200.nc',encoding=encoding)
#
#
#
# elapsed_time = T.time() - start_time
# print(T.strftime("%H:%M:%S", T.gmtime(elapsed_time)))



# output = {}
# output['gray'] = gray
#
# import pickle
#
# with open('wam1572548400_20average_5delt.pickle','wb') as f:
#     pickle.dump(output, f)









#
# import cv2
#
# cap = cv2.VideoCapture(0)# cv2.VideoCapture('wamImageSpaceC2_1572548400_20seconds.avi')
#
# # Mouse function
# def select_point(event, x, y, flags, params):
#     global point, point_selected
#     if event == cv2.EVENT_LBUTTONDOWN:
#         point = (x, y)
#         point_selected = True
#
# #cv2.namedWindow("Frame")
# #cv2.setMouseCallback("Frame")
#
#
# point_selected = False
# point = ()
#
# while True:
#     ret, frame = cap.read()
#
#     if point_selected is True:
#         cv2.circle(frame, point, 5, (0,0,255), 2)
#     #fgmask = fgbg.apply(frame)
#     #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#
#     cv2.imshow("Frame",frame)
#     k = cv2.waitKey(1)
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.createBackgroundSubtractorMOG2()
#
# while(1):
#     ret, frame = cap.read()
#
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

