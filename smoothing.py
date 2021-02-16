import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import cv2
import sys

def getColorExact(localExtrema, ntscIm):
    [n,m,d] = ntscIm.shape
    imgSize = n*m
    nI = np.zeros(ntscIm.shape, dtype=ntscIm.dtype)
    nI[...,0] = ntscIm[...,0]
    indsM = np.arange(imgSize).reshape(n, m)

    wd = 1
    length = 0
    consts_len = 0
    col_inds = np.zeros(imgSize*(2*wd+1)**2, dtype=int)
    row_inds = np.zeros(imgSize*(2*wd+1)**2, dtype=int)
    vals = np.zeros(imgSize*(2*wd+1)**2)
    gvals = np.zeros((2*wd+1)**2)

    for i in range(n):
        for j in range(m):
            if not localExtrema[i,j]:
                tlen = 0
                for ii in range(max(0,i-wd),min(i+wd+1,n)):
                    for jj in range(max(0,j-wd),min(j+wd+1,m)):
                        if(ii != i) or (jj != j):
                            row_inds[length] = consts_len
                            col_inds[length] = indsM[ii,jj]
                            gvals[tlen] = ntscIm[ii,jj,0]
                            length  = length +1
                            tlen = tlen+1
                t_val = ntscIm[i,j,0] # center pixel Y value
                gvals[tlen] = t_val
                c_var = np.mean((gvals[:tlen+1] - np.mean(gvals[:tlen+1]) )**2)
                csig=c_var*0.6
                mgv = min((gvals[:tlen]-t_val)**2)
                if csig < (-mgv/np.log(0.01)):
                    csig = -mgv/np.log(0.01)
                if csig < 0.000002:
                    csig = 0.000002
                gvals[:tlen] = np.exp(-(gvals[:tlen]-t_val)**2/csig)
                gvals[:tlen] = gvals[:tlen]/sum(gvals[:tlen])
                vals[length-tlen:length] = -gvals[:tlen]

            row_inds[length] = consts_len
            col_inds[length] = indsM[i,j]
            vals[length]=1
            length = length+1
            consts_len = consts_len+1

    vals = vals[:length]
    col_inds = col_inds[:length]
    row_inds = row_inds[:length]

    A_csc = csc_matrix((vals, (row_inds, col_inds)), shape=(consts_len, imgSize))
    LU = splu(A_csc)
    b = np.zeros(A_csc.shape[0],dtype=ntscIm.dtype )
    for dim in range(1,d):
        curIm = ntscIm[:,:,dim]
        b[indsM[localExtrema != 0]] = curIm[localExtrema]
        new_vals = LU.solve(b)
        nI[...,dim] = new_vals.reshape((n,m))

    return nI


def EdgePreservingSmooth(I,k=3):
    """
    Implement "Edge-preserving Multiscale Image Decomposition based on Local Extrema"

    Parameters
    -----------
    I: input image( BGR image or grayscale image )
    k: kernel size, default = 3

    Returns
    -----------
    M: smoothed image( BGR image or grayscale image )
    localMax: local maxima extrema( boolean matrix )
    localMin: local minima extrema( boolean matrix )
    MaxEnvelope: extremal envelopes of maxima extrema( Y+ extremal envelopes at each BGR channel )
    MinEnvelope: extermal envelope of minima extrema( Y+ extremal envelopes at each BGR channel )

    """

    # wd: half width of kernel size
    wd = k//2

    if I.ndim == 3:
        channel = I.shape[2]
        YUV = cv2.cvtColor(I,cv2.COLOR_BGR2YUV)
        Y = np.double(YUV[:,:,0])/255
        image = np.double(I)/255
        #cv2.imshow("Y",Y)
        #cv2.waitKey(0)

    else:
        channel = 1
        Y = np.double(I)/255
    print("Extrema location")
    height,width = Y.shape
    localMax = np.zeros( Y.shape, dtype=bool)
    localMin = np.zeros( Y.shape, dtype=bool)
    for i in range(height):
        for j in range(width):
            center = Y[i,j]

            ii_start = max(0,i-wd)
            ii_end = min(i+wd+1,height)
            jj_start = max(0,j-wd)
            jj_end = min(j+wd+1,width)
            cover = Y[ii_start:ii_end,jj_start:jj_end]

            maxcount = np.sum(cover > center)
            mincount = np.sum(center > cover)
            if maxcount <= k-1:
                localMax[i,j] = True
            if mincount <= k-1:
                localMin[i,j] = True

    print("Extermal envelope construction")
    Y_BGR = np.zeros((height,width,4))
    Y_BGR[...,0] = Y;
    for i in range(channel):
        Y_BGR[...,i+1] = image[...,i]

    MaxEnvelope = getColorExact(localMax, Y_BGR)
    MinEnvelope = getColorExact(localMin, Y_BGR)

    print("Computation of the smoothed mean")
    M = (MaxEnvelope[:,:,1:(channel+1)] + MinEnvelope[:,:,1:(channel+1)])/2;
    M = (M*255).astype(np.uint8)
    #cv2.imshow("M",M)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return M, localMax, localMin, MaxEnvelope, MinEnvelope


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Usage:', sys.argv[0], '<ImagePath>', '<KernelSize>', '<Iteration>')
        sys.exit(1)

    imagepath = sys.argv[1]
    kernelsize = int(sys.argv[2])
    iteration = int(sys.argv[3])

    I = cv2.imread(imagepath)
    M= I.copy()
    for i in range(iteration):
        print('Iteration: ', str(i+1))
        M,localmax, localmin, maxenvelope, minenvelope = EdgePreservingSmooth(M,kernelsize)
        kernelsize += 4
        print('')

    I_YUV = cv2.cvtColor(I,cv2.COLOR_BGR2YUV)
    M_YUV = cv2.cvtColor(M,cv2.COLOR_BGR2YUV)
    D = I_YUV[:,:,0]-M_YUV[:,:,0]

    # Make the grey scale image have three channels
    grey_3_channel = cv2.cvtColor(D, cv2.COLOR_GRAY2BGR)
    numpy_horizontal = np.hstack(( I, M, grey_3_channel))
    cv2.imshow('Edge-preserving Smooth Result', numpy_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

