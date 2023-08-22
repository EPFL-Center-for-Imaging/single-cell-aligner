"""
2023-08-22 Edward Andò EFPL Center for Imaging

Tool to align images of a cell rotating all over the place
→ N.B. assumes that the displacement has already been corrected by masking and cropping

https://www.youtube.com/watch?v=xysbNbvUDZA 

Inputs:
  - 2D Timeseries of cell rotating
  - 2D timeseries of mask used to correlate only what we want

Output:
  - registered image

Usage:
  python 03-incrementalRegistration.py grey.tif mask.tif
"""

step = 40

import sys
import numpy
import scipy.ndimage
import spam.DIC
import tifffile
from tqdm import tqdm

# Load 2D timeseries
im = tifffile.imread(sys.argv[1])
# Load mask
mask = tifffile.imread(sys.argv[2])

assert im.shape[0] == mask.shape[0], "Images and Mask have different time-lengths!"

imStep = im[::step]
maskStep = mask[::step] # HACK!


PhisIncrementalStep = numpy.zeros((imStep.shape[0],4,4))
RSsStep = numpy.zeros(imStep.shape[0])


###############################################################
## Step 1: Measure and incremental motion between stepped frames
###############################################################
print(f"1/2: Doing incremental registrations with step = {step}")
for t in tqdm(range(1,imStep.shape[0])):
    # Backward correlation we that we're moving t-1 onto t!
    reg = spam.DIC.register(
        imStep[t],
        imStep[t-1],
        im1mask=maskStep[t],
        imShowProgress=0,
        verbose=0,
        maxIterations=150,
        PhiRigid=True
    )
    PhisIncrementalStep[t] = reg['Phi']
    RSsStep[t] = reg['returnStatus']

print(numpy.unique(RSsStep, return_counts=True))

###############################################################
## Step 2: Add up all the measured increments into a total movement
###############################################################
print(f"2/2: Adding up increments into a total motion and applying it to all timesteps")
PhisIncrementalStep[0] = numpy.eye(4)
PhiCurrent = numpy.eye(4)
#Rots = numpy.zeros(imStep.shape[0])
#TransYX = numpy.zeros((imStep.shape[0], 2))
PhisTotal = numpy.zeros((im.shape[0],4,4))

imOut = numpy.zeros_like(im)


# Loop over steps
for T in tqdm(range(0,imStep.shape[0])):
    PhiIncrementDecomp = spam.deformation.decomposePhi(PhisIncrementalStep[T])
    # Loop inside steps to gradually apply the measured motion
    for t in range(step):
        globalTimeIndex = step*T+t
        if globalTimeIndex >= im.shape[0]:
            break
        else:
            scale = ((t+1)/step)
            PhiTemp = spam.deformation.computePhi(
            {
                't': [0, PhiIncrementDecomp['t'][0]*scale, PhiIncrementDecomp['t'][1]*scale],
                'r': [PhiIncrementDecomp['r'][0]*scale, 0, 0]
            }
            )
            PhiCurrentInterp = numpy.dot(PhiCurrent, PhiTemp)
            imOut[globalTimeIndex] = spam.DIC.applyPhiPython(im[globalTimeIndex], Phi=PhiCurrentInterp)

    #PhiCurrentDecomp = spam.deformation.decomposePhi(PhiCurrent)
    #print(f"t={PhiCurrentDecomp['t'][1:3]}, r={PhiCurrentDecomp['r'][0]}")
    PhiCurrent = numpy.dot(PhiCurrent, PhisIncrementalStep[T])

tifffile.imwrite(f"{sys.argv[1][0:-4]}-registered.tif", imOut)

################################################################
### Step 3: Upscale and apply to ALL timesteps
################################################################
#print(f"3/3: Interpolating motion and applying to whole timeseries")

##Rots = scipy.ndimage.zoom(Rots, step, order=1)
##TransYX = scipy.ndimage.zoom(TransYX, step, order=1)

#imOut = numpy.zeros_like(im)
#for t in tqdm(range(0,im.shape[0])):
    #Phi = spam.deformation.computePhi(
        #{
            #'t': [0, TransYX[t,0], TransYX[t,1]],
            #'r': [Rots[t], 0, 0]
        #}
    #)
    #imOut[t] = spam.DIC.applyPhiPython(im[t], Phi=Phi)
