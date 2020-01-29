

import argusIO
import time as T
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr


#data = xr.open_dataset('rawForTestWAM.nc')
data = xr.open_dataset('merged1572548400.nc')

merged = data['merged'].values
x = data['xFRF'].values
y = data['yFRF'].values



timex = np.nanmean(merged,axis=2)

m,n,p = np.shape(merged)

smoothtimex = np.zeros((m,n,p),dtype='uint8')

timeInt = 25
#merged = np.uint8((merged-np.min(merged))/(np.max(merged)-np.min(merged)))

for i in range(p-timeInt*2):
    smoothtimex[:,:,i] = np.nanmean(merged[:,:,i:i+(timeInt*2)],axis=2)


#
#
smoothtimex = smoothtimex[:,:,:-(2*timeInt)]


smooth_timex = np.nanmean(smoothtimex,axis=2)
smooth_darkest = np.nanmin(smoothtimex,axis=2)
smooth_brightest = np.nanmax(smoothtimex,axis=2)+1

subtracted = smoothtimex-smooth_darkest[:,:,np.newaxis]

divided = np.zeros((np.shape(smoothtimex)))
m,n,p = np.shape(smoothtimex)
for i in range(p):
    divided[:,:,i] = subtracted[:,:,i]/smooth_brightest

#multiplier = 254/smooth_darkest[:,:]
multiplied = divided*255

subset = multiplied[:1500,:,:]
#
# #
# #
# # darkest = np.nanmin(smoothtimex,axis=2)
# # brightest = np.nanmax(smoothtimex,axis=2)
# # standard = np.nanstd(smoothtimex,axis=2)
# # #test = (cuttimex[:,:,100]-timex) # + 37)*(255/76)
# # #test2 = -(cuttimex[:,:,100]-timex)
# # #negatives = np.where((test2<0))
# # #test2[negatives] = 0
# #
# # #tempTest = (smoothtimex[:,:,1000])/standard
# #
# # diff = np.diff(smoothtimex,axis=2)
#
#subset = smoothtimex[:1500,:,:]
nn,nm,pp = np.shape(subset)
import cv2
frame_width = nm
frame_height = nn
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
out2 = cv2.VideoWriter('wam30sAverage_0pt5DT_testThresh.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height),0)

for i in range(pp):

    temp = subset[:,:,i]#diff #((smoothtimex[:,:,i])/standard)*10

    # blurred = cv2.bilateralFilter(temp.astype(np.uint8), 15, 225, 225)
    # ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # waveBreak = np.where((thresh < 250))
    # noneBreak = np.where((thresh > 250))
    # altered = temp.copy()
    # altered[waveBreak] = np.median(altered[noneBreak])


    frame = np.flipud(np.uint8(temp))

    #frame = np.flipud(np.uint8((temp+(np.abs(np.min(temp))))*(255/(np.max(temp)-np.min(temp)))))
    out2.write(frame)

out2.release()

# Closes all the frames
cv2.destroyAllWindows()
#
#
# fig, ax = plt.subplots(3,1)
#
# rown = 205
#
# ax[0].plot(smoothtimex[:,rown-10,0])
# ax[1].plot(smoothtimex[:,rown-5,0])
# ax[2].plot(smoothtimex[:,rown,0])
# #ax[3].plot(smoothtimex[:,rown,0])
# #ax[4].plot(smoothtimex[:,rown+1,0])
# #ax[5].plot(smoothtimex[:,rown+2,0])
#
# ax[0].plot(smoothtimex[:,rown-10,20])
# ax[1].plot(smoothtimex[:,rown-5,20])
# ax[2].plot(smoothtimex[:,rown,20])
# #ax[3].plot(smoothtimex[:,rown,20])
# #ax[4].plot(smoothtimex[:,rown+1,20])
# #ax[5].plot(smoothtimex[:,rown+2,20])
#
# ax[0].plot(smoothtimex[:,rown-10,40])
# ax[1].plot(smoothtimex[:,rown-5,40])
# ax[2].plot(smoothtimex[:,rown,40])
# #ax[3].plot(smoothtimex[:,rown,40])
# #ax[4].plot(smoothtimex[:,rown+1,40])
# #ax[5].plot(smoothtimex[:,rown+2,40])
#
# from scipy.signal import butter, lfilter
#
# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
#
#
# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y
#
#
# n,m,p = np.shape(smoothtimex)
# fs = 2
# T = p/2
# nsamples = T * fs
# t = np.linspace(0, T, nsamples, endpoint=False)
#
# from scipy import fftpack
# sig = smoothtimex[260,230,:]
#
# sig_fft = fftpack.fft(sig)
# power = np.abs(sig_fft)
# sample_freq = fftpack.fftfreq(sig.size, d=2)
#
# plt.figure()
# plt.plot(t,sig)
#
# plt.figure(figsize=(6, 5))
# plt.plot(sample_freq, power)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('plower')
#
# # Find the peak frequency: we can focus on only the positive frequencies
# #pos_mask = np.where((sample_freq > 0.0026) & (sample_freq < 0.0036))
# pos_mask = np.where((sample_freq > 0.00))
#
# freqs = sample_freq[pos_mask]
# peak_freq = freqs[power[pos_mask].argmax()]
#
# high_freq_fft = sig_fft.copy()
# high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
# high_freq_fft[np.abs(sample_freq) < peak_freq] = 0
#
# filtered_sig = fftpack.ifft(high_freq_fft)
#
#
# plt.figure(figsize=(6, 5))
# plt.plot(t, sig, label='Original signal')
# plt.plot(t, filtered_sig, linewidth=3, label='Filtered signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
#
# plt.legend(loc='best')
#
# # Check that it does indeed correspond to the frequency that we generate
# # the signal with
# #np.allclose(peak_freq, 1./period)
#
# # # An inner plot to show the peak frequency
# # axes = plt.axes([0.55, 0.3, 0.3, 0.5])
# # plt.title('Peak frequency')
# # plt.plot(freqs[:8], power[:8])
# # plt.setp(axes, yticks=[])
#
#
# # import the necessary packages
# import numpy as np
# import argparse
# import glob
# import cv2
#
#
# def auto_canny(image, sigma=0.33):
#     # compute the median of the single channel pixel intensities
#     v = np.median(image)
#
#     # apply automatic Canny edge detection using the computed median
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = cv2.Canny(image, lower, upper)
#
#     # return the edged image
#     return edged
#
#
#
# image = smoothtimex[:1300,290,100]
# #blurred = cv2.GaussianBlur(image, (7,7),0)
# blurred = cv2.bilateralFilter(image, 15, 225, 225)
# ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# waveBreak = np.where((thresh<250))
# altered = blurred.copy()
# altered[waveBreak] = 0
# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
#
#
#
# # apply Canny edge detection using a wide threshold, tight
# # threshold, and automatically determined threshold
# wide = cv2.Canny(blurred, 10, 200)
# tight = cv2.Canny(blurred, 225, 250)
# auto = auto_canny(blurred, sigma=0.05)
#
# # show the images
# cv2.imshow("Original+blurred", np.hstack([image,blurred]))
# cv2.imshow("Edges", np.hstack([wide, tight, auto]))
#


#
#
# # Sample rate and desired cutoff frequencies (in Hz).
# n,m,p = np.shape(smoothtimex)
# fs = 2
# T = p/2
# nsamples = T * fs
#
# lowcut = 0.012
# highcut = 0.018
# t = np.linspace(0, T, nsamples, endpoint=False)
# x = smoothtimex[300,200,:]
#
# # a = 0.02
# # f0 = 600.0
# # x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
# # x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
# # x += a * np.cos(2 * np.pi * f0 * t + .11)
# # x += 0.2 * np.cos(2 * np.pi * 2000 * t)
# plt.figure(2)
# plt.clf()
# plt.plot(t, x, label='Noisy signal')
#
# y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
# plt.plot(t, y)#, label='Filtered signal (%g Hz)' % f0)
# plt.xlabel('time (seconds)')
# #plt.hlines([-a, a], 0, T, linestyles='--')
# plt.grid(True)
# plt.axis('tight')
# plt.legend(loc='upper left')
# plt.ylim([-100,300])
# plt.show()

#
#
# from openpiv import tools, process, validation, filters, scaling
#
#
# frame_a = cuttimex[:,:,1000]-timex
# frame_b = cuttimex[:,:,1010]-timex
#
# fig,ax = plt.subplots(1,2)
# ax[0].imshow(frame_a,cmap=plt.cm.gray)
# ax[1].imshow(frame_b,cmap=plt.cm.gray)
#
# winsize = 8 # pixels
# searchsize = 16  # pixels, search in image B
# overlap = 4 # pixels
# dt = 0.5 # sec
#
# u0, v0, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
#
# x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )
# u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3 )
# u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=10, kernel_size=2)
# x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 96.52 )
# tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )
# tools.display_vector_field('exp1_001.txt', scale=10, width=0.0025)
#
#
# plt.quiver(x,y,u3,v3)
