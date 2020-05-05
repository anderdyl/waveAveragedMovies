import argusIO
import time as T
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr


def vBar(stack,ystack,xstack):

    y = np.append([0],[(ystack-ystack[0])])

    Twin = 64
    N = len(ystack)
    M = Twin
    dt = 0.5
    dy = np.abs(np.mean(np.diff(y)))
    L = dy*N
    T = dt*M
    taper = np.outer(np.bartlett(M), np.bartlett(N))    # creating taper window
    vB = [-3, 3]
    dv = 0.05
    v = np.arange(vB[0],vB[1]+.01,dv)
    sigma = 0.075
    UB = [np.inf, np.max(vB), 2, np.inf]    # upper bounds on search
    LB = [0, np.min(vB), .01, 0]            # lower bounds on search

    # make a k vector
    if (N % 2) == 0:
        k = np.arange(-N/2, (N/2-1)+1, 1)/L
    else:
        k = np.arange((-(N-1)/2), ((N-1)/2)+1, 1)/L

    gk = np.where((k>0) & (k<(1/(2*dy))))

    k = k[gk]
    # make a f vector
    if (M % 2) == 0:
        f = np.arange((M/2), -(M/2-1)-1, -1)/T
    else:
        f = np.arange(((M-1)/2), (-(M-1)/2)-1, -1)/T

    gf = np.where((np.abs(f)>0) & (np.abs(f)<(1/(2*dt))))
    f = f[gf]

    fkB = [np.inf, 0, np.min(k)*2, 2]

    K = np.tile(k[:,np.newaxis].T,[len(f),1])
    F = np.tile(f[:,np.newaxis], [1, len(k)])

    FKind = np.where((np.abs(F)<fkB[0]) & (np.abs(F)>fkB[1]) & (np.abs(K)>fkB[2]) & (np.abs(K)<fkB[3]))

    Smask = np.nan*np.ones(np.shape(K))
    Smask[FKind[0],FKind[1]] = 1

    fkny = [np.nanmax(np.abs(F[np.where((np.isreal(Smask)))])), np.nanmax(np.abs(K[np.where((np.isreal(Smask)))]))]

    # ideally we will step through blocks...
    block = stack[0:M,:]
    meanStack2 = np.mean(np.mean(block))
    block = stack[0:M,:]-np.tile(np.mean(block,axis=0), (M,1))

    p95 = np.percentile(block, 95)
    p50 = np.percentile(block, 50)
    QCspan = p95-p50

    stxfft = np.fft.fft2(block*taper)

    S = 2*stxfft*np.conj(stxfft)/(len(k)*Twin)
    S = np.fft.fftshift(S)
    gkm, gfm = np.meshgrid(gk, gf)
    S = S[gfm, gkm]*Smask

    from scipy.interpolate import interp1d

    Sv = np.zeros((len(v), len(k)))
    for ii in range(len(k)):
        #print(ii)
        fun = interp1d(f/k[ii], S[:,ii], fill_value = 0, bounds_error=False)
        Sv[:,ii] = fun(v)
        del fun

    V = np.nansum(Sv.T, axis=0)
    V2 = np.append([0, 0, 0], V)
    V3 = np.append(V2,[0, 0])

    def running_mean(x, N):
        cumsum = np.cumsum(x) #np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    out = running_mean(V3,5)
    if np.nanmax(out) == 0:
        print('Well shit, we''re dividing by a zero')
        output = dict()
        output['meanV'] = np.nan
        output['stdV'] = np.nan
        output['meanX'] = np.mean(xstack)
        output['meanY'] = np.mean(ystack)
        output['beta'] = np.nan
        output['residuals'] = np.nan
        output['resnorm'] = np.nan
        output['chi2'] = np.nan
        output['prob'] = np.nan
        output['QCspan'] = np.nan
        output['SE'] = np.nan
        output['ci'] = np.nan
        output['cispan'] = np.nan
        output['SNR'] = np.nan
        output['V3'] = V3
        output['Sv'] = Sv

        # output['prob'] =
        # plotVbar(block,y,S,f,k,Sv,v,V4,fitted,popt)
        return output

    V4 = out/np.nanmax(out)

    maxv = np.nanmax(V4)
    maxvind = np.where(V4 == np.nanmax(V4))

    mdV = v[maxvind]

    gind = np.where((np.real(V4)))




    beta0 = [maxv, np.double(mdV), 0.25, np.nanmean(V4)]
    jv = v[gind]

    from scipy.optimize import curve_fit

    def getVint(jv, fkny):

        fnyq = fkny[0]
        knyq = fkny[1]
        fL = 1/32
        Vint = np.nan*np.ones((np.shape(jv)))
        vLowInd = np.where(np.abs(jv) <= fnyq/knyq)
        vHiInd = np.where(np.abs(jv) > fnyq/knyq)
        Vint[vLowInd] = (np.square(knyq))/2
        Vint[vHiInd] = (np.square(fnyq))/(2*np.square(jv[vHiInd]))
        Vint = Vint/np.max(Vint)

        return Vint

    Vint = getVint(jv,fkny)

    X = np.vstack((jv,Vint))

    def func(X,a,b,c,d):
        vfun, Vfun = X
        return a*np.exp(-np.square((vfun-b)/c)) + d*Vfun



    xdata = np.vstack((jv,Vint))
    #y = func(beta0, jv, fkny)
    ydata = V4[gind]
    try:
        popt, pcov = curve_fit(func, xdata, ydata, beta0, bounds=(LB,UB))
        fitted = func(X, popt[0], popt[1], popt[2], popt[3])

        residuals = ydata-func(X, popt[0], popt[1], popt[2], popt[3])
        resnorm = np.sum(np.square(residuals))
        chiSquare = resnorm/np.square(sigma)
        from scipy.stats import chi2
        #print(len(gind[0])-len(beta0))
        prob = 1.0 - chi2.cdf(chiSquare, len(gind[0])-len(beta0))

        SE = np.sqrt(np.diag(np.abs(pcov)))*1.97  # std. dev. x 1.96 -> 95% conf
        ci = [popt[1]-SE[1], popt[1]+SE[1]]
        cispan = SE[1]*2
        SNR = popt[0]/popt[3]

    except:
        print("Unable to fit optimal velocity function: returning Nans")
        popt = np.zeros((3,))
        popt[1] = np.nan
        popt[2] = np.nan
        residuals = np.nan
        resnorm = np.nan
        chiSquare = np.nan
        prob = np.nan
        SE = np.nan
        ci = np.nan
        cispan = np.nan
        SNR = np.nan
        v = np.nan
        fitted = np.nan
        V4 = np.nan



    def plotVbar(block,y,S,f,k,Sv,v,V4,fitted,popt):
        fig, axs = plt.subplots(2,2)
        axs[0,0].imshow(block,origin='upper', extent=[y[0], y[-1], len(block)/2, 0])
        axs[0,0].set(xlabel='Alongshore (m)', ylabel='Time (s)')

        im1 = axs[0,1].imshow(np.fliplr(np.log10(np.abs(S.T))), origin='lower', extent=[f[0], f[-1], k[0], k[-1]])
        axs[0,1].set(xlabel='f (Hz)', ylabel='$k_{y}$ (1/m)')
        fig.colorbar(im1, ax=axs[0,1])

        im2 = axs[1,0].imshow((np.log10(np.abs(Sv.T))), origin='lower', extent=[v[0], v[-1], k[0], k[-1]])
        axs[1,0].set_aspect(2.5)
        axs[1,0].set(xlabel='velocity (m/s)', ylabel='$k_{y}$ (1/m)')
        fig.colorbar(im2, ax=axs[1,0])

        axs[1,1].plot(v,V4,label='S(v)')
        axs[1,1].plot(v,fitted,'r--',label='$S_{model}$(v)')
        axs[1,1].plot([0, 0]+popt[1],[0, 1],'k')
        axs[1,1].plot([popt[2],popt[2]]+popt[1],[0, 1],'k--')
        axs[1,1].plot([-popt[2],-popt[2]]+popt[1],[0, 1],'k--')
        axs[1,1].set(xlabel='velocity (m/s)', ylabel='spectral density')
        axs[1,1].set_title('{:.4f} m/s'.format(popt[1]))
        axs[1,1].set_ylim([0, 1])
        axs[1,1].legend()


    output = dict()
    output['meanV'] = popt[1]
    output['stdV'] = popt[2]
    output['meanX'] = np.mean(xstack)
    output['meanY'] = np.mean(ystack)
    output['beta'] = popt
    output['residuals'] = residuals
    output['resnorm'] = resnorm
    output['chi2'] = chiSquare
    output['QCspan'] = QCspan
    output['prob'] = prob
    output['SE'] = SE
    output['ci'] = ci
    output['cispan'] = cispan
    output['SNR'] = SNR
    output['v'] = v
    output['V4'] = V4
    output['fitted'] = fitted

    #output['prob'] =
    #plotVbar(block,y,S,f,k,Sv,v,V4,fitted,popt)
    return output





def radonCurrent(stackIn,timeIn,ystack,xstack):



    # stackIn: input stack (nt,nx)
    # timeIn: time input (second since start of stack) ... ie (0...2048)
    # xy: x and y pixal location of data of stack (size=(nx,2))
    # tWin: temporal window to do analysis over (in pixel points)
    # tStep: temporal step to include (not currently used)

    xy = np.vstack((ystack, xstack)).T
    radialFilterThresh = 15
    tWin = 120
    plotFlag = 0
    fnameOutBase = 'radonOCMout'

    dt = np.median(np.diff(timeIn))
    dx = np.median(np.diff(xy[:, 0]))

    pdx = 1  # spatial resolution degradation (in pixel points) --- sub sampling of image
    freq_x = 1 / (dx * pdx)
    Wx = (np.max(xy[:, 0]) - np.min(xy[:, 0])) / dx

    pdt = 1  # temporal resolution degradation (in pixel points)
    freq_t = 1 / (dt * pdt)  # temporal frequency (1/dt) (taking into account pdt)
    iang = np.arange(1, 181, 1)

    M = stackIn

    tt = 0

    MR = M[tt:(tt + tWin), :]
    MRtime = timeIn[tt:(tt + tWin)]
    nt = np.shape(MR)[0]

    from skimage.transform import radon
    from scipy.signal import detrend
    MRdetrend = detrend(MR)
    sinogram = radon(image=MRdetrend, theta=iang, circle=False)

    nr = np.shape(sinogram)[0]
    amp = nr / nt
    k = nt
    trk = np.floor((np.shape(sinogram)[1] / 2)) - np.floor(
        (0 * np.cos(iang * np.pi / 180) + ((np.shape(sinogram)[1]) / 2) - k * amp) * np.sin(iang * np.pi / 180))

    trk = trk - np.min(trk)
    res = (nt * dt) / (trk * 2)

    r2 = sinogram.copy()

    def moving_average(a, n=21):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def movmean(a, n):
        y_padded = np.pad(a, (n // 2, n - 1 - n // 2), mode='edge')
        y_smooth = np.convolve(y_padded, np.ones((n,)) / n, mode='valid')
        return y_smooth

    for i in range(len(iang)):
        r2[:, i] = sinogram[:, i] - movmean(a=sinogram[:, i], n=int(np.round(1 + radialFilterThresh / res[i])))

    #AngPixlIntensDensity = np.std(r2[int(np.round(np.shape(r2)[0] / 4)):int(3 * np.shape(r2)[0] / 4), :], axis=0)
    AngPixlIntensDensity = np.std(r2,axis=0)
    a2 = np.argmax(AngPixlIntensDensity)

    # if len(freq_x) == 1:
    #    C2 = (1/np.mean(freq_x))/(np.tan((90-a2)*np.pi/180)*(1/np.mean(freq_t)))
    # else:
    c2 = (1 / np.mean(freq_x)) / (np.tan((90 - a2) * np.pi / 180) * (1 / np.mean(freq_t)))
    from skimage.transform import iradon
    #invR = iradon(r2[int(np.round(np.shape(r2)[0] / 4)):int(3 * np.shape(r2)[0] / 4), :], theta=iang, circle=False)
    invR = iradon(r2, theta=iang, circle=False)
    nMR, mMR = np.shape(MR)
    nInv, mInv = np.shape(invR)

    if nMR == nInv:
        if mMR == mInv:
            invR = invR
        else:
            invR = invR[:, int(mMR-np.round(mMR/2)):int(mMR+np.round(mMR/2))]
    else:
        invR = invR[int(nMR-np.round(nMR/2)):int(nMR+np.round(nMR/2)), :]




    if plotFlag == 1:
        fig = plt.figure(figsize=(10, 4))
        ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=1, rowspan=4)

        imR1 = ax1.imshow(MR, extent=[np.max(ystack), np.min(ystack), MRtime[-1], MRtime[0]])
        plt.colorbar(imR1, ax=ax1)
        ax1.set_title('Raw image stack')
        ax1.set_xlabel('Alongshore (m)')
        ax1.set_ylabel('time (s)')
        ax2 = plt.subplot2grid((4, 3), (0, 1), colspan=1, rowspan=4)

        imR2 = ax2.imshow(invR, extent=[np.max(ystack), np.min(ystack), MRtime[-1], MRtime[0]])
        plt.colorbar(imR2, ax=ax2)
        ax2.set_title('Filtered image stack')
        ax2.set_xlabel('Alongshore (m)')
        ax2.set_ylabel('time (s)')
        ax3 = plt.subplot2grid((4, 3), (0, 2), colspan=2, rowspan=1)
        line1 = ax3.plot(iang, AngPixlIntensDensity)
        ax3.set_title('Mean Celerity = {:.2f}'.format(c2))
        ax4 = plt.subplot2grid((4, 3), (1, 2), colspan=2, rowspan=3)

        imR4 = ax4.imshow(r2, extent=[np.min(iang), np.max(iang), -(np.shape(sinogram)[0] - np.shape(sinogram)[0] / 2),
                                      (np.shape(sinogram)[0] - np.shape(sinogram)[0] / 2)])
        plt.colorbar(imR4, ax=ax4)
        ax4.set_title('filtered radon transform')
        ax4.set_xlabel('angle')
        ax4.set_ylabel('radial distance')

    output = dict()
    output['MR'] = MR
    output['invR'] = invR
    output['MRtime'] = MRtime
    output['iang'] = iang
    output['AngPixlIntensDensity'] = AngPixlIntensDensity
    output['C2'] = c2
    output['sinogram'] = sinogram
    output['R2'] = r2

    return output
