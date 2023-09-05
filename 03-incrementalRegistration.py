"""
2023-08-22 Edward Andò EFPL Center for Imaging

Tool to align images of a cell rotating all over the place
→ N.B. assumes that the displacement has already been corrected by masking and cropping

Centipede reference for metachronal wave:
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
import os.path
import scipy.interpolate
from tqdm import tqdm
import matplotlib.pyplot as plt

import matplotlib
font = {'family' : '',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)

# Load 2D timeseries
im = tifffile.imread(sys.argv[1])
# Load mask
mask = tifffile.imread(sys.argv[2])

assert im.shape[0] == mask.shape[0], "Images and Mask have different time-lengths!"

imStep = im[::step]
maskStep = mask[::step]

PhisTotalStepName = f"{sys.argv[1][0:-4]}-PhisTotal-step{step}.npy"
PhisIncrementalStepName = f"{sys.argv[1][0:-4]}-PhisIncremental-step{step}.npy"

PhisIncrementalStep = numpy.zeros((imStep.shape[0],4,4))
RSsStep = numpy.zeros(imStep.shape[0])

#if os.path.isfile(PhisTotalStepName) and os.path.isfile(PhisIncrementalStepName):
if os.path.isfile(PhisTotalStepName):
    PhisTotalStep = numpy.load(PhisTotalStepName)
    #PhisIncrementalStep = numpy.load(PhisIncrementalStepName)

else:
    ###############################################################
    ## Step 1: Measure and incremental motion between stepped frames
    ###############################################################
    print(f"1/3: Doing incremental registrations with step = {step}")
    for T in tqdm(range(1,imStep.shape[0])):
        # Backward correlation we that we're moving T-1 onto T!
        reg = spam.DIC.register(
            imStep[T],
            imStep[T-1],
            im1mask=maskStep[T],
            imShowProgress=0,
            verbose=0,
            returnPhiMaskCentre=False,
            maxIterations=150,
            PhiRigid=True
        )
        PhisIncrementalStep[T] = reg['Phi']
        RSsStep[T] = reg['returnStatus']

    print(numpy.unique(RSsStep, return_counts=True))

    ###############################################################
    ## Step 2: Add up all the measured increments into a total movement
    ###############################################################
    print(f"2/3: Adding up increments into a total motion")
    PhisIncrementalStep[0] = numpy.eye(4)
    PhiCurrent = numpy.eye(4)

    PhisTotalStep = numpy.zeros_like(PhisIncrementalStep)

    # First of all just compute the total Phi for all the measured increments ("steps")
    for T in tqdm(range(0,imStep.shape[0])):
        PhiCurrent = numpy.dot(PhiCurrent, PhisIncrementalStep[T])
        PhisTotalStep[T] = PhiCurrent

    numpy.save(PhisTotalStepName, PhisTotalStep)
    numpy.save(PhisIncrementalStepName, PhisIncrementalStep)


#translationTotalStepSpline = scipy.interpolate.
plt.plot(
    numpy.arange(PhisTotalStep.shape[0])*step,
    PhisTotalStep[:,1,-1],
    'x-',
    label='y-disp'
)
plt.plot(
    numpy.arange(PhisTotalStep.shape[0])*step,
    PhisTotalStep[:,2,-1],
    'x-',
    label='x-disp'
)
plt.xlabel(f'time (steps of {step})')
plt.ylabel('px displacement')
plt.legend()
#plt.show()

# Create and initialise PhisTotal
PhisTotal = numpy.zeros((im.shape[0], 4, 4))
for T in range(PhisTotal.shape[0]):
    PhisTotal[T] = numpy.eye(4)

# Set up splines on raw Phi components?
# Curvilinear coordinates are [0, 1] in the STEP basis, careful!
tck, _ = scipy.interpolate.splprep(
    [
        PhisTotalStep[:, 1, 1],
        PhisTotalStep[:, 1, 2],
        PhisTotalStep[:, 1, 3],
        PhisTotalStep[:, 2, 1],
        PhisTotalStep[:, 2, 2],
        PhisTotalStep[:, 2, 3],
    ],
    s=10,
    k=3,
)

# What does the time range [0, 1] for the spline (well really just the one)
#   correspond to in the global?
# t = 1 means "step" time of imStep.shape[0]-1 which is a real time of step*(imStep.shape[0]-1)
splineTimes = numpy.arange(0, im.shape[0])/(step*(imStep.shape[0]-1))

interpolatedPhiTotal = scipy.interpolate.splev(splineTimes, tck)
PhisTotal[:, 1, 1] = interpolatedPhiTotal[0]
PhisTotal[:, 1, 2] = interpolatedPhiTotal[1]
PhisTotal[:, 1, 3] = interpolatedPhiTotal[2]
PhisTotal[:, 2, 1] = interpolatedPhiTotal[3]
PhisTotal[:, 2, 2] = interpolatedPhiTotal[4]
PhisTotal[:, 2, 3] = interpolatedPhiTotal[5]

plt.plot(
    splineTimes*(step*(imStep.shape[0]-1)),
    PhisTotal[:,1,-1],
    '-',
    label='y-disp (interp)'
)
plt.plot(
    splineTimes*(step*(imStep.shape[0]-1)),
    PhisTotal[:,2,-1],
    '-',
    label='x-disp (interp)'
)
plt.show()

#exit()



###############################################################
## Step 3: apply to whole-time-series
###############################################################
print(f"3/3: Applying to all time steps")
imOut = numpy.zeros_like(im)
# Loop over steps
for t in tqdm(range(0,im.shape[0])):
    #PhiIncrementDecomp = spam.deformation.decomposePhi(PhisIncrementalStep[T])
    ## Loop inside steps to gradually apply the measured motion
    #for t in range(step):
        #globalTimeIndex = step*T+t
        #if globalTimeIndex >= im.shape[0]:
            #break
        #else:
            #scale = ((t+1)/step)
            #PhiTemp = spam.deformation.computePhi(
            #{
                #'t': [0, PhiIncrementDecomp['t'][0]*scale, PhiIncrementDecomp['t'][1]*scale],
                #'r': [PhiIncrementDecomp['r'][0]*scale, 0, 0]
            #}
            #)
            #PhiCurrentInterp = numpy.dot(PhiCurrent, PhiTemp)
            #imOut[globalTimeIndex] = spam.DIC.applyPhiPython(im[globalTimeIndex], Phi=PhiCurrentInterp)

    #PhiCurrent = numpy.dot(PhiCurrent, PhisIncrementalStep[T])
    imOut[t] = spam.DIC.applyPhiPython(im[t], Phi=PhisTotal[t])


tifffile.imwrite(f"{sys.argv[1][0:-4]}-registered-{step}step.tif", imOut)
