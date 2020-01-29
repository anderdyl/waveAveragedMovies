


import argusIO
import time as T
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr
from ocmIO import radonCurrent
from ocmIO import vBar
import cv2


file = 'merged1572548400.nc'
data = xr.open_dataset(file)
merged = data['merged'].values
x = data['xFRF'].values
y = data['yFRF'].values
imTime = data['t'].values


yi = 780
xi = 260
te = 120

imYind = np.where((y >= yi) & (y <= (yi + 30)))
imXind = np.where((x == xi))

ystack = y[imYind]
xstack = x[imXind] * np.ones((np.shape(ystack)))
imSeconds = np.arange(0, te/2, 0.5)

stack = np.fliplr(merged[imYind[0], imXind[0], 0:te].T)
bad = np.argwhere(np.isnan(stack))
stack[bad] = np.nanmean(stack)


stackOG = stack


# Histogram equalization
stackHE = cv2.equalizeHist(stack.astype('uint8'))
# Laplacian
stackLP = cv2.Laplacian(stack.astype('uint8'),cv2.CV_8U,ksize=3)
# Sobel X gradient
stackSX = cv2.Sobel(stack.astype('uint8'), cv2.CV_8U, 1, 0, ksize=3)
stackSY = cv2.Sobel(stack.astype('uint8'), cv2.CV_8U, 0, 1, ksize=3)

# stack = cv2.Sobel(stack.astype('float64'),cv2.CV_64F,1,0,ksize=5)
# stack = np.uint8(np.absolute(stack))

test2 = radonCurrent(stack, imSeconds, ystack[0:-1], xstack[0:-1])

test = vBar(stack, ystack, xstack)
# test = vBar(test2['invR'], ystack[0:-1], xstack[0:-1])



fig = plt.figure(figsize=(15, 4))
ax1 = plt.subplot2grid((4, 5), (0, 0), colspan=1, rowspan=4)
ax1.imshow((stackOG), vmin=0, vmax=255, extent=[ystack[-1], ystack[0], te/2, 0])
ax1.set_aspect(0.75)
ax1.set_xlabel('yFRF (m)')
ax1.set_ylabel('Time (s)')
ax1.set_title('Original')
#ax1.plot(insituDrift['yDrift'], insituDrift['seconds'], '.')
#ax1.set_title('X = {}'.format(xloops[xc]))

ax2 = plt.subplot2grid((4, 5), (0, 1), colspan=1, rowspan=4)
ax2.imshow((stackHE), vmin=0, vmax=255, extent=[ystack[-1], ystack[0], te/2, 0])
ax2.set_aspect(0.75)
ax2.set_xlabel('yFRF (m)')
ax2.set_ylabel('Time (s)')
ax2.set_title('Histogram Equalization')
#ax2.set_title('Median V = {:.2f} m/s'.format(np.round(100 * insituDrift['vMean']) / 100))

ax3 = plt.subplot2grid((4, 5), (0, 2), colspan=1, rowspan=4)
ax3.imshow((stackSX), vmin=0, vmax=255, extent=[ystack[-1], ystack[0], te/2, 0])
ax3.set_aspect(0.75)
ax3.set_xlabel('yFRF (m)')
ax3.set_ylabel('Time (s)')
ax3.set_title('Horizontal Gradient Filter')

ax4 = plt.subplot2grid((4, 5), (0, 3), colspan=1, rowspan=4)
ax4.imshow((stackSY), vmin=0, vmax=255, extent=[ystack[-1], ystack[0], te/2, 0])
ax4.set_aspect(0.75)
ax4.set_xlabel('yFRF (m)')
ax4.set_ylabel('Time (s)')
ax4.set_title('Vertical Gradient Filter')

ax5 = plt.subplot2grid((4, 5), (0, 4), colspan=1, rowspan=4)
ax5.imshow(stackLP, vmin=0, vmax=255, extent=[ystack[-1], ystack[0], te/2, 0])
#ax5.imshow((test2['invR']), vmin=0, vmax=255, extent=[ystack[-1], ystack[0], te/2, 0])
ax5.set_aspect(0.75)
ax5.set_xlabel('yFRF (m)')
ax5.set_ylabel('Time (s)')
ax5.set_title('Laplacian Filter')

# ax3.plot(test['v'], test['V4'], label='S(v)')
# ax3.plot(test['v'], test['fitted'], 'r--', label='$S_{model}$(v)')
# ax3.plot([0, 0] + test['meanV'], [0, 1], 'k')
# ax3.plot([test['stdV'], test['stdV']] + test['meanV'], [0, 1], 'k--')
# ax3.plot([-test['stdV'], -test['stdV']] + test['meanV'], [0, 1], 'k--')
# ax3.set(xlabel='velocity (m/s)', ylabel='spectral density')
# ax3.set_title('Vbar mean = {:.2f} m/s'.format(test['meanV']))
# ax3.set_ylim([0, 1])
# ax3.legend()
#
# ax4a = plt.subplot2grid((4, 5), (0, 3), colspan=1, rowspan=1)
# line1 = ax4a.plot(test2['iang'], test2['AngPixlIntensDensity'])
# ax4a.set_title('Mean Radon Celerity = {:.2f}'.format(test2['C2']))
#
# ax4 = plt.subplot2grid((4, 5), (1, 3), colspan=1, rowspan=3)
# imR4 = ax4.imshow(test2['R2'], extent=[np.min(test2['iang']), np.max(test2['iang']),
#                                        -(np.shape(test2['sinogram'])[0] - np.shape(test2['sinogram'])[0] / 2),
#                                        (np.shape(test2['sinogram'])[0] - np.shape(test2['sinogram'])[0] / 2)])
# plt.colorbar(imR4, ax=ax4)
# ax4.set_title('filtered radon transform')
# ax4.set_xlabel('angle')
# ax4.set_ylabel('radial distance')
#
# ax5 = plt.subplot2grid((4, 5), (0, 4), colspan=1, rowspan=4)
#
# imR2 = ax5.imshow(test2['invR'], extent=[np.max(ystack), np.min(ystack), test2['MRtime'][-1], test2['MRtime'][0]])
# plt.colorbar(imR2, ax=ax5)
# ax5.set_title('Filtered image stack')
# ax5.set_xlabel('Alongshore (m)')
# ax5.set_ylabel('time (s)')
# ax5.set_aspect(0.75)

plt.tight_layout()




