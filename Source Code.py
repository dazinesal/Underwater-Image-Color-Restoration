# This program is usable in any Python inclusive environment.
# The user must install all libraries cited below before running the program.
# The user must change the folder and name of the source picture through the variable “folder” and “source” declared in the main function below.
# The processed images will be saved in the same folder where the input image exists.


import cv2
import heapq
import numpy as np
import scipy.ndimage as nd
from cv2.ximgproc import guidedFilter



def dark_channel(i):
    M, N, _ = i.shape #Dimensions of the image array

    #Padding an array with the values of i
    padded = np.pad(i, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')

    #Creating the dark channel array
    dark = np.zeros((M, N))

    #Selecting the lowest intensity pixels out of the three channels of i
    for i, j in np.ndindex(dark.shape):
        dark[i, j] = np.min(padded[i:i + w, j:j + w, :])
    
    return dark

def bright_channel(i):
    M, N, _ = i.shape #Dimensions of the image array

    #Padding an array with the values of i
    padded = np.pad(i, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')

    #Creating the bright channel array
    bright = np.zeros((M, N))

    #Selecting the highest intensity pixels out of the three channels
    for i, j in np.ndindex(bright.shape):
        bright[i, j] = np.max(padded[i:i + w, j:j + w, :])
    
    return bright

def bgsubr(i, bright):
    M, N = bright.shape #Dimensions of the image array

    #Getting the indexes of the maximum, medium and minmum color channels
    cmax, cmid, cmin, _,_,_,_,_,_ = channel_intensities(image)

    #Creating the maximum color difference array
    bgsubr = np.zeros((M, N))

    #Seprating i into the three color channels accordingly
    arrcmax = i[...,cmax]
    arrcmid = i[...,cmid]
    arrcmin = i[...,cmin]

    #Calculating the maximum channel difference in each pixel
    for mi in range(M):
        for ni in range(N):
            bgsubr[mi][ni] = 1 - max(max(arrcmax[mi][ni]-arrcmin[mi][ni], 0), max(arrcmid[mi][ni]-arrcmin[mi][ni],0))

    return bgsubr

def rectify_bright(bgsubr):

    #Calculating the saturation channel of the image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    #Calculating the coefficient lambda
    lambd = (hsv[...,1].max())/255

    #Calculating the rectified bright channel
    ibright = (bright*lambd) + (bgsubr*(1-lambd))

    return ibright

def atmospheric_light(i, ibright):
    M, N = ibright.shape #Dimensions of the rectified bright channel array
    at = np.empty(3) #Atmospheric light
    selectvar = [] #Array used to get the variance of the darkest pixels 

    #Storing the 3D input array into 2D array
    flati = i.reshape(M*N, 3)

    #Extracting the gray filter image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255

    #Producing the variance within a block for each pixel from gray image
    win_var = nd.generic_filter(gray, np.var, size = 3)
    minvar=256

    #Finding the top 1%*M*N darkest pixels
    flatbright = ibright.ravel()
    top = heapq.nsmallest(int(M*N*0.1),flatbright)

    #Finding the dark pixel with the miminmum variance intensity
    a = np.where(np.isin(ibright, top))
    for n in range (0, len(a[0])):
        (b,c) = (a[0][n], a[1][n])
        selectvar.append(win_var [b,c])
        if (minvar>np.amin(selectvar)):
            minvar = np.amin(selectvar)
            ib, ic = b,c
            if(minvar == 0): break

    #Getting the atmospheric light intensity
    at[0] = i[ib,ic,0]
    at[1] = i[ib,ic,1]
    at[2] = i[ib,ic,2]

    return at

def initial_transmission(a, ibright):
    M, N = ibright.shape #Dimensions of the image array
    init = np.zeros((M,N)) #Initial transmittance

    #Calulating the transmittance over each channel
    for i in range(3):
        init = init + ((ibright-a[i])/(1.-a[i]))
        init = (init - np.min(init))/(np.max(init) - np.min(init))
    init = init/3

    #Calculating the average value of the transmittance
    return (init - np.min(init))/(np.max(init) - np.min(init))


def guidedFilter(gray,init):
    #Bluring the guide image using the box filter
    mean_gray = cv2.boxFilter(gray,cv2.CV_64F,(60,60))

    #Bluring the input image using the box filter
    mean_init = cv2.boxFilter(init, cv2.CV_64F,(60,60))

    #Mean of guide * input
    mean_gi = cv2.boxFilter(gray*init,cv2.CV_64F,(60,60))

    #Covariance of guide and input
    cov_gi = mean_gi - (mean_gray*mean_init)

    #Mean of guide*guide
    mean_gg = cv2.boxFilter(gray*gray,cv2.CV_64F,(60,60))

    #variance of guide
    var_g  = mean_gg - (mean_gray*mean_gray)

    #Calculate the mean 
    a = cov_gi/(var_g + 0.0001)
    b = mean_init - a*mean_gray

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(60,60))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(60,60))

    refined = mean_a*gray + mean_b

    return refined

def refined_transmission(init):
    refined = np.full_like(init, 0)

    #Extracting the gray filter image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255


    refined = guidedFilter(gray, init)

    return refined


def channel_intensities(image):
    
    #Reading the 3 channels of the image
    b, g, r = cv2.split(image)
    
    #Calculating the mean intensity of each channel
    t = image.size / 3
    bx = float(np.sum(b)) / t
    gx = float(np.sum(g)) / t
    rx = float(np.sum(r)) / t

    #Identifying the indexes of the maximum, medium and minimum channel intensities
    var={bx:bi,gx:gi,rx:ri}
    cmax=var.get(max(var))
    cmin=var.get(min(var))
    if ((cmax==1 or cmax==2) and (cmin==1 or cmin==2)):
        cmid =0
    if ((cmax==0 or cmax==2) and (cmin==0 or cmin==2)):
        cmid =1
    if ((cmax==0 or cmax==1) and (cmin==0 or cmin==1)):
        cmid =2

    return cmax,cmid,cmin, bx,gx,rx, b,g,r

def restoration_image(i, a, refined):
    M, N, _ = i.shape

    #Broadcasting the refined transmission into a 3D array
    corrected = np.broadcast_to(refined[:,:,None], (refined.shape[0], refined.shape[1], 3))

    #Restoring the original image
    j = ((i-a)/corrected) + a

    return j

def histogram_equalization(j):
    M, N, _ = j.shape #Dimensions of the restored image

    #Creating the means of the color channels
    bluemean, greenmean, redmean = float("%0.5f" % (2)), float("%0.5f" % (2)) ,float("%0.5f" % (2))

    #Handling the case of negative pixels
    for mi in range(M):
        for ni in range(N):
            if (j[mi,ni,0] <= 0):
                j[mi,ni,0] = 0
            if (j[mi,ni,1] <= 0):                
                j[mi,ni,1] = 0
            if (j[mi,ni,2] <= 0):
                j[mi,ni,2] = 0

    #Getting the means and arrays of each channel
    _, _, _, b,g,r, barr,garr,rarr = channel_intensities(j*255)
    
    #Converting the intensity range to [0,1]
    barr=barr/255
    garr=garr/255
    rarr=rarr/255

    #Assigning the wanted means from each channel
    bidx=0.5
    gidx=0.5
    ridx=0.5

    #Another alternative to assigning the wanted means from each channel
    #bidx=((b/255)+0.5)/2
    #gidx=((g/255)+0.5)/2
    #ridx=((r/255)+0.5)/2



    #Equalizing the blue channel
    if (bidx>0):
        bint = float("%0.5f" % (bidx))
        while bluemean != bint:
            bluemean = float("%0.5f" % (float((np.sum(barr))) / (M*N)))
            powb = np.log(bint)/np.log(bluemean)
            barr = (barr) ** (powb)

    #Equalizing the green channel
    if (gidx>0):
        gint = float("%0.5f" % (gidx))
        while greenmean != gint:
            greenmean = float("%0.5f" % (float((np.sum(garr))) / (M*N))) 
            powg = np.log(gint)/np.log(greenmean)
            garr = (garr) ** (powg)

    #Equalizing the red channel
    if (ridx>0):
        rint = float("%0.5f" % (ridx))
        while redmean != rint:
            redmean = float("%0.5f" % (float((np.sum(rarr))) / (M*N)))
            powr = np.log(rint)/ np.log(redmean)
            rarr = (rarr) ** (powr)

    #Combining the three channels into the new restored image
    for mi in range(M):
        for ni in range(N):
                j[mi,ni,0]=barr[mi,ni]
                j[mi,ni,1]=garr[mi,ni]                
                j[mi,ni,2]=rarr[mi,ni]

    return j






if __name__ == '__main__':

    global w,bi,gi,ri, folder, source, image
    w= 15 #Window size
    bi,gi,ri=0,1,2 #Color channels indexes

    #Reading image
    print ('Reading Image...')
    folder = "C:/Users/Salmane/Documents/" #Folder of reading/writing
    source = "jpp.jpeg" #Name of the file
    image = cv2.imread(folder + source)

    #Converting image into numpy array with 3 channels
    i=np.asarray(image, dtype=np.float64)
    i=i[:, :, :3]/255

    #Priting the image dimensions
    height, width, _ = i.shape
    print ('Image dimensions: ( Height:', height, ', Width:', width, ')')
    if (height >600) and (width >600):
        print('Your source image is large. The program might risk running for a long time.')
    print('\n')  

    #Writing the input image
    cv2.imwrite(folder + 'original_image.jpg', image)  


    #Getting the bright channel
    print ('Processing bright channel...')
    bright=bright_channel(i)
    cv2.imwrite(folder + 'bright_channel.jpeg', bright*255)
    print('Bright channel saved successfully')
    print('\n')


    #Getting the bright channel
    print ('Procession the bright channel rectification...')
    bgsubr = bgsubr(i, bright)
    ibright = rectify_bright(bgsubr)
    cv2.imwrite(folder + 'rectified_bright.jpeg', ibright*255)
    print('Rectified bright channel saved successfully')
    print('\n')

    #Getting the atmospheric light
    print ('Processing atmospheric light...')
    a = atmospheric_light(i, ibright)
    print ('Atmospheric light: {}'.format(a))
    print('\n')

    #Getting the initial transmission
    print ('Processing initial transmission...')
    init = initial_transmission(a, ibright)
    white = np.full_like(bright, 255)
    cv2.imwrite(folder + 'initial_transmission.jpeg', init*white)
    print('Initial transmission saved successfully')
    print('\n')

    #Getting the refined transmission
    print ('Processing refined transmission...')
    refined = refined_transmission(init)
    cv2.imwrite(folder + 'transmission_corrected.jpeg', refined*white)
    print('Refined transmission saved successfully')
    print('\n')

    #Getting the restored image
    print ('Processing image restoration...')
    j = restoration_image(i, a, refined)
    cv2.imwrite(folder + 'restored_image.jpeg', j*255)
    print('Restored image saved successfully')
    print('\n')


    #Getting the histogram equalization
    print ('Processing histogram equalization...')
    result = histogram_equalization(j)
    cv2.imwrite(folder + 'final_result.jpeg', result*255)
    print('Final result saved successfully')
    print('\n')