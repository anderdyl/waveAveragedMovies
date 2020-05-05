

import pickle
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from objMapPrep import coarseBackground
from objMapPrep import binMorph
from obj_map_interp import map_interp
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


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        #>>> # linear interpolation of NaNs
        #>>> nans, x= nan_helper(y)
        #>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


file = '/media/dylananderson/LaCie/netcdfs/merged1571770800.nc'
data = xr.open_dataset(file)
merged = data['merged'].values
x = data['xFRF'].values
y = data['yFRF'].values
imTime = data['t'].values

ymins = 500
ymaxs = 1300
maxcolory = 0.8
scaleQ = 8
vBarMovie = 'oct30vbarVariability.avi'

dbfile = open('spatialPickle1571770800_60s_40m_30soverlaps.pickle', 'rb')
data = pickle.load(dbfile)
#for keys in data:
#    print(keys, '=>', data[keys])
dbfile.close()


flowPickle = open('oct22wamFlowFields.pickle','rb')
flow = pickle.load(flowPickle)
flowPickle.close()


# output['allFlow'] = allFlow
# output['allFlowX'] = allFlowX
# output['allFlowY'] = allFlowY
# output['OGxgrid'] = OGxgrid
# output['OGygrid'] = OGygrid
# output['timexIM'] = prev_gray
# output['avgFX'] = timexX
# output['avgFY'] = timexY





nn, mm, pp = np.shape(data['vvVbar'])

trustedV = np.nan * np.ones((nn, mm, pp))

#for i in range(pp):
for i in range(pp):

    cispan5 = data['ogvBarCispan'][:,:,i].copy()
    theBad1 =  np.where(cispan5 > .18)

    chisq = data['ogvBarChi2'][:,:,i].copy()
    theBad2 =  np.where(chisq > 325)

    qcspan = data['ogvBarQCspan'][:,:,i].copy()
    theBad3 =  np.where(qcspan < 8)


    vvBARconfident = data['ogvvVbar'][:,:,i].copy()
    vvBARconfident[theBad2] = np.nan
    vvBARconfident[theBad1] = np.nan
    vvBARconfident[theBad3] = np.nan

    meanConfident = np.nanmean(vvBARconfident)
    stdConfident = np.nanstd(vvBARconfident)
    negV = data['ogvvVbar'][:,:,i].copy()   #vvBARconfident.copy()
    theBad3 = np.where(negV > (meanConfident+2*stdConfident))
    theBad4 = np.where(negV < (meanConfident-2*stdConfident))

    vvBARconfident[theBad3] = np.nan
    vvBARconfident[theBad4] = np.nan

    trustedV[:,:,i] = vvBARconfident
    from scipy import interpolate
    # option 1
    # xx = data['xx'].copy().flatten()
    # yy = data['yy'].copy().flatten()
    # vv = vvBARconfident.copy().flatten()
    #
    # good = np.where(np.isfinite(vv))
    # f = interpolate.interp2d(xx[good], yy[good], vv[good], kind='cubic',bounds_error=False)
    #
    # surf = f(data['xx'][:,0], data['yy'][0,:])

    # # option 2
    # xx = data['xx'].copy().flatten()
    # yy = data['yy'].copy().flatten()
    # vv = vvBARconfident.copy().flatten()
    # good = np.where(np.isfinite(vv))
    # points = np.vstack((xx[good], yy[good])).T
    # xi = np.vstack((xx,yy)).T

    #xnew, ynew = np.meshgrid(np.arange(125,275,10),np.arange(545,975,10))

    #surf = interpolate.griddata(points, vv[good], (data['xx'], data['yy']), method='linear')
    #surf = interpolate.griddata(points, vv[good], (xnew, ynew), method='linear')


    #
    # fig = plt.figure(figsize=(9, 6))
    # ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=2)
    # h1 = ax1.imshow(np.flipud(merged[:, :, 50]), cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[1], y[-1]])
    # ax1.set_xlabel('xFRF (m)')
    # ax1.set_ylabel('yFRF (s)')
    # # c2 = ax1.pcolor(data['xx'],data['yy'],data['vvVbar'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # # c2 = ax1.pcolor(data['xx'],data['yy'],-data['vvDrifter'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    #
    # c2 = ax1.pcolor(data['xx'], data['yy'], vvBARconfident, vmin=-1.5, vmax=1.5, cmap='RdBu', )
    # # c2 = ax1.pcolor(xnew, ynew, surf, vmin=-1.5, vmax=1.5, cmap='RdBu', )
    #
    # plt.colorbar(c2, ax=ax1)
    # ax1.set_title('Raw Return from vBar')
    # ax1.set_ylim([ymins, ymaxs])
    #
    # ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=2)
    # h2 = ax2.imshow(np.flipud(merged[:, :, 50]), cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[1], y[-1]])
    # ax2.set_xlabel('xFRF (m)')
    # ax2.set_ylabel('yFRF (s)')
    # #    c2 = ax1.pcolor(data['xx'],data['yy'],data['vvVbar'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # # c2 = ax1.pcolor(data['xx'],data['yy'],-data['vvDrifter'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # c3 = ax2.pcolor(data['xx'], data['yy'], data['vBarChi2'][:,:,i], vmin=0, vmax=3000)#, vmin=0, vmax=100)
    # plt.colorbar(c3, ax=ax2)
    # ax2.set_title('Confidence')
    # ax2.set_ylim([ymins, ymaxs])

    # ax3 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=2)
    # h3 = ax3.imshow(np.flipud(merged[:, :, 50]), cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[0], y[-1]])
    # ax3.set_xlabel('xFRF (m)')
    # ax3.set_ylabel('yFRF (s)')
    # #    c2 = ax1.pcolor(data['xx'],data['yy'],data['vvVbar'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # # c2 = ax1.pcolor(data['xx'],data['yy'],-data['vvDrifter'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # c4 = ax3.pcolor(data['xx'], data['yy'], surf, vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # plt.colorbar(c4, ax=ax3)
    # ax3.set_title('Interpolating')
    # ax3.set_ylim([500, 1100])

    #plt.tight_layout()


#     fig = plt.figure(figsize=(4,8))
#     ax1 = fig.add_subplot(111)
#     h1 = ax1.imshow(np.flipud(merged[:,:,50]), cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[0], y[-1]])
#     ax1.set_xlabel('xFRF (m)')
#     ax1.set_ylabel('yFRF (s)')
# #    c2 = ax1.pcolor(data['xx'],data['yy'],data['vvVbar'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
#    # c2 = ax1.pcolor(data['xx'],data['yy'],-data['vvDrifter'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
#     c2 = ax1.pcolor(data['xx'],data['yy'],vvBARconfident, vmin=-1.5, vmax=1.5, cmap='RdBu',)



import matplotlib.colors

minfy = -maxcolory#np.max(np.abs(np.nanmean(trustedV,axis=2)))/2
maxfy = maxcolory#np.max(np.abs(np.nanmean(trustedV,axis=2)))/2
norm3 = matplotlib.colors.Normalize(vmin=minfy, vmax=maxfy)



fig = plt.figure(figsize=(6, 10))
ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
h1 = ax1.imshow(np.flipud(np.nanmean(merged,axis=2))/1.5, cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[1], y[-1]])
ax1.set_xlabel('xFRF (m)')
ax1.set_ylabel('yFRF (s)')
# c2 = ax1.pcolor(data['xx'],data['yy'],data['vvVbar'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
# c2 = ax1.pcolor(data['xx'],data['yy'],-data['vvDrifter'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)

#c2 = ax1.pcolor(data['xx'], data['yy'], vvBARconfident, vmin=-1.5, vmax=1.5, cmap='RdBu', )
c2 = ax1.quiver(data['xx'],data['yy'],0*data['xx'],-np.nanmean(trustedV,axis=2),-np.nanmean(trustedV,axis=2),scale=scaleQ,cmap='RdBu_r',width=0.005,norm=norm3)
# c2 = ax1.pcolor(xnew, ynew, surf, vmin=-1.5, vmax=1.5, cmap='RdBu', )

fig.colorbar(c2, ax=ax1)
ax1.set_title('17 minute average')
ax1.set_ylim([ymins, ymaxs])
plt.tight_layout()





### can we make some simple profiles

yind = np.where((data['yy'] > 880) & (data['yy'] < 940))

xsub = np.mean(np.reshape(data['xx'][yind], (40,3)).T,axis=0)
ysub = np.mean(np.reshape(data['yy'][yind], (40,3)).T,axis=0)
avgV = np.nanmean(trustedV,axis=2)
vsub = np.mean(np.reshape(avgV[yind], (40,3)).T,axis=0)


avgFx = flow['avgFX'].copy()
avgFy = flow['avgFY'].copy()
avgX = flow['OGxgrid'].copy()
avgY = flow['OGygrid'].copy()

flowind = np.where((avgY > 920) & (avgY < 980))

fxsub = np.nanmean(np.reshape(avgX[flowind],(59,350)),axis=0)
fysub = np.nanmean(np.reshape(avgY[flowind],(59,350)),axis=0)
fvsub = np.nanmean(np.reshape(avgFy[flowind],(59,350)),axis=0)


fig = plt.figure()
ax10 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
ax10.plot(xsub,-vsub,label='timestack/vbar')
ax10.plot(fxsub,-fvsub/4,label='wam/optical flow')
ax10.set_xlabel('xFRF (m)')
ax10.set_ylabel('longshore current (m/s)')
ax10.legend()
#ax11 = plt.subplot2grid((1,2),(0,1),rowspan=1,colspan=1)
#ax11.plot(fxsub,fvsub)

SDF




for i in range(pp):



    fig = plt.figure(figsize=(14, 10))


    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)
    h1 = ax1.imshow(np.flipud(merged[:, :, i*30]) / 1.5, cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[1], y[-1]])
    ax1.set_xlabel('xFRF (m)')
    ax1.set_ylabel('yFRF (s)')
    # c2 = ax1.pcolor(data['xx'],data['yy'],data['vvVbar'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # c2 = ax1.pcolor(data['xx'],data['yy'],-data['vvDrifter'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)

    # c2 = ax1.pcolor(data['xx'], data['yy'], vvBARconfident, vmin=-1.5, vmax=1.5, cmap='RdBu', )
    #c2 = ax1.quiver(data['xx'], data['yy'], 0 * data['xx'], -np.nanmean(trustedV, axis=2),
    #                -np.nanmean(trustedV, axis=2), scale=12, cmap='RdBu_r', width=0.005, norm=norm3)
    # c2 = ax1.pcolor(xnew, ynew, surf, vmin=-1.5, vmax=1.5, cmap='RdBu', )

    #fig.colorbar(c2, ax=ax1)
    ax1.set_title('Merged Snaps')
    ax1.set_ylim([ymins, ymaxs])


    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
    h2 = ax2.imshow(np.flipud(np.nanmean(merged[:, :, (i*30-28):i*30],axis=2)) / 1.5, cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[1], y[-1]])
    ax2.set_xlabel('xFRF (m)')
    ax2.set_ylabel('yFRF (s)')
    # c2 = ax1.pcolor(data['xx'],data['yy'],data['vvVbar'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # c2 = ax1.pcolor(data['xx'],data['yy'],-data['vvDrifter'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)

    # c2 = ax1.pcolor(data['xx'], data['yy'], vvBARconfident, vmin=-1.5, vmax=1.5, cmap='RdBu', )
    if i < 4:
        c2 = ax2.quiver(data['xx'], data['yy'], 0 * data['xx'], -np.nanmean(trustedV[:,:,0:i], axis=2),
                        -np.nanmean(trustedV[:,:,0:i], axis=2), scale=scaleQ, cmap='RdBu_r', width=0.005, norm=norm3)
    else:
        c2 = ax2.quiver(data['xx'], data['yy'], 0 * data['xx'], -np.nanmean(trustedV[:,:,i-4:i], axis=2),
                        -np.nanmean(trustedV[:,:,i-4:i], axis=2), scale=scaleQ, cmap='RdBu_r', width=0.005, norm=norm3)
    # c2 = ax1.pcolor(xnew, ynew, surf, vmin=-1.5, vmax=1.5, cmap='RdBu', )

    #fig.colorbar(c2, ax=ax2)
    ax2.set_title('2 min Time Average')
    ax2.set_ylim([ymins, ymaxs])


    ax3 = plt.subplot2grid((1, 3), (0, 2), colspan=1, rowspan=1)
    h3 = ax3.imshow(np.flipud(np.nanmean(merged,axis=2)) / 1.5, cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[1], y[-1]])
    ax3.set_xlabel('xFRF (m)')
    ax3.set_ylabel('yFRF (s)')
    # c2 = ax1.pcolor(data['xx'],data['yy'],data['vvVbar'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)
    # c2 = ax1.pcolor(data['xx'],data['yy'],-data['vvDrifter'][:,:,i], vmin=-1.5, vmax=1.5, cmap='RdBu',)

    # c2 = ax1.pcolor(data['xx'], data['yy'], vvBARconfident, vmin=-1.5, vmax=1.5, cmap='RdBu', )
    c3 = ax3.quiver(data['xx'], data['yy'], 0 * data['xx'], -np.nanmean(trustedV, axis=2),
                    -np.nanmean(trustedV, axis=2), scale=scaleQ, cmap='RdBu_r', width=0.005, norm=norm3)
    # c2 = ax1.pcolor(xnew, ynew, surf, vmin=-1.5, vmax=1.5, cmap='RdBu', )

    #fig.colorbar(c3, ax=ax3)
    ax3.set_title('Cumulative Average')
    ax3.set_ylim([ymins, ymaxs])

    plt.tight_layout()
    if i < 10:
       plt.savefig('/home/dylananderson/projects/drifters/vBarMovie1/frame00'+str(i)+'.png')
    elif i < 100:
       plt.savefig('/home/dylananderson/projects/drifters/vBarMovie1/frame0'+str(i)+'.png')
    else:
       plt.savefig('/home/dylananderson/projects/drifters/vBarMovie1/frame'+str(i)+'.png')
    plt.close()



# import os
# geomorphdir = '/home/dylananderson/projects/drifters/vBarMovie1'
#
# files = os.listdir(geomorphdir)
#
# files.sort()
#
# files_path = [os.path.join(geomorphdir,x) for x in os.listdir(geomorphdir)]
#
# files_path.sort()
#
# import cv2
#
# frame = cv2.imread(files_path[0])
# height, width, layers = frame.shape
# forcc = cv2.VideoWriter_fourcc(*'XVID')
# video = cv2.VideoWriter(vBarMovie, forcc, 4, (width, height))
# for image in files_path:
#     video.write(cv2.imread(image))
# cv2.destroyAllWindows()
# video.release()


#
# trustedV = trustedV
#
#
# meanTrust = np.nanmean(trustedV,axis=2)
# bad = np.where(np.isnan(meanTrust))
# meanTrust[bad] = np.nanmean(meanTrust)
# xdata = (data['xx'].flatten()) #[:,0]
# ydata = (data['yy'].flatten()) #[0,:]
# #xn = data['xx'][:,0]
# #yn = data['yy'][0,:]
# cdx = 15
# cdy = 10
#
# xc, yc, zc, xn, yn = coarseBackground(x=xdata, y=ydata, z=meanTrust.flatten(), cdx=cdx, cdy=cdy)
#
# plt.figure()
# ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
# h1 = ax1.imshow(np.flipud(merged[:, :, 50]), cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[0], y[-1]])
# h2 = ax1.pcolor(xc,yc,zc,vmin=-1.2,vmax=1.2,cmap='RdBu')
# ax1.set_xlabel('xFRF (m)')
# ax1.set_ylabel('yFRF (s)')
#
#
#
# import numpy.ma as ma
# xs = ma.masked_array(xdata.astype('float'), mask = np.zeros((len(xdata),)))
# ys = ma.masked_array(ydata.astype('float'), mask = np.zeros((len(xdata),)))
# zs = ma.masked_array(trustedV[:,:,0].flatten(), mask = np.zeros((len(xdata),)))
#
# binned = binMorph(xn, yn, xs, ys, zs)
#
# countBinThresh = 2
# bc = binned['binCounts']
# id = np.nonzero(bc > countBinThresh)
# bcvals = bc[id]
# zbin = binned['zBinVar']
# # check standard error to see if noise value you chose is reasonable
# stdErr = np.sqrt(zbin[id]/bc[id])
# zbM = binned['zBinMedian']
# zFluc = zbM[id] - zc[id]
#
#
# Lx = 25    # meters cross-shore
# Ly = 50    # meters alongshore
#
# # map example survey
# # extract relevant domain for mapping from coarser grid scale
# xmin = np.min(xc[id])-Lx*3  # min cross-shore
# xmax = np.max(xc[id])+Lx*3  # max cross-shore
# ymin = np.min(yc[id])-Ly*3  # min alongshore
# ymax = np.max(yc[id])+Ly*3  # max alongshore
# idInt = np.where((xc > xmin) & (xc < xmax) & (yc > ymin) & (yc < ymax))
#
#
# dgcov, dcovE, A, Aprime, mapFluc, nmseEst, dcovA, dcovA2, sigVar = map_interp(x=xc[id], y=yc[id], zFluc=zFluc, noise=0.01, Lx=Lx, Ly=Ly, xInt=xc[idInt], yInt=yc[idInt])
# #dgcov, dcovE, A, Aprime, mapFluc, nmseEst, dcovA, dcovA2, sigVar = map_interp(x=xc[id], y=yc[id], zFluc=zFluc, noise=noise, Lx=Lx, Ly=Ly, xInt=xx[idInt], yInt=yy[idInt])
#
# allzeros = np.ones(xc.shape)
# nmseest = np.ones(xc.shape)
# nmseest[idInt] = nmseEst.T
# mapfluc = np.zeros(xc.shape)
# mapfluc[idInt] = mapFluc
# goodi = np.nonzero(nmseest < .2)
# badi = np.nonzero(nmseest > .2)
# mapfluc[badi] = allzeros[badi]
#
# zc = zc
#
# mapz = mapfluc+zc
#
# # look at error estimates
# fig7, ax7 = plt.subplots(1, 1, figsize=(5, 5))
# sc7 = ax7.pcolor(xc, yc, nmseest)
# cbar = plt.colorbar(sc7, ax=ax7)
# #cbar.set_label('elevation [m')
# ax7.set_title('NMSE estimate from map of example survey')
#
#
# fig7, ax7 = plt.subplots(1, 1, figsize=(5, 5))
# sc7 = ax7.pcolor(xc, yc, mapfluc)
# cbar = plt.colorbar(sc7, ax=ax7)
# #cbar.set_label('elevation [m')
# #ax7.set_title('NMSE estimate from map of example survey')
#
# # look at error estimates
# fig8, ax8 = plt.subplots(1, 1, figsize=(5, 5))
# sc8 = ax8.pcolor(xc, yc, mapfluc)
# cbar = plt.colorbar(sc8, ax=ax8)
# #cbar.set_label('elevation [m')
# ax8.set_title('mapped elevation fluctuation example survey')
#
# # look at error estimates
# fig9, ax9 = plt.subplots(1, 1, figsize=(5, 5))
# sc8 = ax9.scatter(xc[goodi], yc[goodi], c=mapfluc[goodi])
# cbar = plt.colorbar(sc8, ax=ax9)
# #cbar.set_label('elevation [m')
# ax9.set_title('mapped elevation fluctuation (only good values))')
#
# # look at the final map produced
# fig10, ax10 = plt.subplots(1, 1, figsize=(5, 5))
# sc10 = ax10.pcolor(xc, yc, mapz,cmap='RdBu')
# cbar = plt.colorbar(sc10, ax=ax10)
# #cbar.set_label('elevation [m')
# ax10.set_title('final mapped elevation example survey [m]')
#



#
# theBad = np.argwhere(data['vBarCispan'][:,:,0] > .4)
#
# cispan5 = data['vBarCispan'][:,:,0].copy()
# theBad2 =  np.where(cispan5 > .15)
#
# vvBARconfident = data['vvVbar'][:,:,0].copy()
#
# vvBARconfident[theBad2] = np.nan #* np.ones(len(theBad),)
# h3 = ax[1].imshow(np.flipud(merged[:,:,50]), cmap='gray', vmin=0, vmax=255, extent=[x[0], x[-1], y[0], y[-1]])
# ax[1].set(xlabel='xFRF (m)', ylabel='yFRF (s)')
# c3 = ax[1].pcolor(data['xx'],data['yy'],vvBARconfident, vmin=-1.5, vmax=1.5, cmap='RdBu',)
# fig.colorbar(c3, ax=ax[1])
# ax[1].set_title('Filtered Return')



#with open('drifterPickle.pickle', 'rb') as f:
#    test = pickle.load(f)
    # xxDrifter = pickle.load(f)
    # pickle.dump(xxDrifter, f)
    # pickle.dump(yyDrifter, f)
    # pickle.dump(vvDrifter, f)
    # pickle.dump(vvVBAR, f)
    # pickle.dump(vBarCispan, f)
    # pickle.dump(vvRADON, f)
    # pickle.dump(ccDrifter, f)
    # pickle.dump(xloops, f)
    # pickle.dump(yloops, f)