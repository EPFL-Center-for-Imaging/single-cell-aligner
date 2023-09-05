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

step = 20
GRAPH=1
PHI_TIME_FILTER = 0


import sys
import numpy
import scipy.ndimage
import spam.DIC
import tifffile
import os.path
import scipy.interpolate
from tqdm import tqdm

# Load 2D timeseries
im = tifffile.imread(sys.argv[1])
# Load mask
mask = tifffile.imread(sys.argv[2])

assert im.shape[0] == mask.shape[0], "Images and Mask have different time-lengths!"

imStep = im[::step]
maskStep = mask[::step]

PhisTotalStepName = f"{sys.argv[1][0:-4]}-PhisTotal-step{step}.npy"

PhisIncrementalStep = numpy.zeros((imStep.shape[0],4,4))
RSsStep = numpy.zeros(imStep.shape[0])

#if os.path.isfile(PhisTotalStepName) and os.path.isfile(PhisIncrementalStepName):
if os.path.isfile(PhisTotalStepName):
    print(f"Found previous 'PhiTotal' numpy array: {PhisTotalStepName} reloading it and skipping correlation...\n")
    PhisTotalStep = numpy.load(PhisTotalStepName)

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
            # imShowProgress=1,
            # verbose=1,
            updateGradient=1,
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


## We could filter the Phis down the time axis?
if PHI_TIME_FILTER > 0:
    PhisTotalStep = scipy.ndimage.gaussian_filter(
        PhisTotalStep,
        sigma=(PHI_TIME_FILTER,0,0)
    )


# Create and initialise PhisTotal
PhisTotal = numpy.zeros((im.shape[0], 4, 4))
for T in range(PhisTotal.shape[0]):
    PhisTotal[T] = numpy.eye(4)

### Interpolate a value of PhiTotal for each PhiTotalStep
coordinatesInitial = numpy.ones((3, 4*4*im.shape[0]), dtype="<f4")
coordinates_mgrid = numpy.mgrid[0 : im.shape[0], 0 : 4, 0 : 4]

# Copy into coordinatesInitial
coordinatesInitial[0, :] = coordinates_mgrid[0].ravel()/step
coordinatesInitial[1, :] = coordinates_mgrid[1].ravel()
coordinatesInitial[2, :] = coordinates_mgrid[2].ravel()

PhisTotal = scipy.ndimage.map_coordinates(
    PhisTotalStep,
    coordinatesInitial,
    order=3,
    mode='nearest',
).reshape(-1,4,4)

# Impose 2D consistancy that might have numerical noise due to interpolation
PhisTotal[:,0,:]  = [1,0,0,0]
PhisTotal[:,:,0]  = [1,0,0,0]
PhisTotal[:,-1,:] = [0,0,0,1]

if GRAPH:
    import matplotlib.pyplot as plt

    import matplotlib
    font = {'family' : '',
            'weight' : 'normal',
            'size'   : 12}
    matplotlib.rc('font', **font)

    plt.plot(
        numpy.arange(PhisTotalStep.shape[0])*step,
        PhisTotalStep[:,1,-1],
        'rx',
        label='y-disp'
    )
    plt.plot(
        numpy.arange(PhisTotalStep.shape[0])*step,
        PhisTotalStep[:,2,-1],
        'bx',
        label='x-disp'
    )

    plt.plot(
        numpy.arange(PhisTotal.shape[0]),
        PhisTotal[:,1,-1],
        'r-',
        label='y-disp interp'
    )
    plt.plot(
        numpy.arange(PhisTotal.shape[0]),
        PhisTotal[:,2,-1],
        'b-',
        label='x-disp interp'
    )
    plt.xlabel(f'time')
    plt.ylabel('px displacement')
    plt.legend()
    plt.show()

###############################################################
## Step 3: apply to whole-time-series
###############################################################
print(f"3/3: Applying to all time steps")
imOut = numpy.zeros_like(im)
# Loop over steps
for t in tqdm(range(0,im.shape[0])):
    imOut[t] = spam.DIC.applyPhiPython(im[t], Phi=PhisTotal[t], interpolationOrder=1)

tifffile.imwrite(f"{sys.argv[1][0:-4]}-registered-step{step}.tif", imOut)
