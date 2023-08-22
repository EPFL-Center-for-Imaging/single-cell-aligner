"""
2023-08-22 Edward Andò

in a 2D timeseries and accompanying mask, Extract a fixed square crop

usage:
    python 02-maskCropping.py greyscales.tif mask.tif
"""
import numpy
import tifffile
import scipy.ndimage
import spam.helpers
from tqdm import tqdm
import sys

# Size of output square, odd number recommended
squareSize = 251
squareHalfSize = (squareSize-1)//2

greyFile = sys.argv[1]
maskFile = sys.argv[2]

grey = tifffile.imread(greyFile)
mask = tifffile.imread(maskFile) > 0

greyOut = numpy.zeros((grey.shape[0], squareSize, squareSize), dtype=grey.dtype)

for t in tqdm(range(grey.shape[0])):
    COMyx = numpy.round(scipy.ndimage.center_of_mass(mask[t])).astype(int)
    #print(COMyx)

    greyOut[t] = spam.helpers.slicePadded(
        grey[t][numpy.newaxis, ...],
        [0, 1, COMyx[0]-squareHalfSize, COMyx[0]+squareHalfSize+1, COMyx[1]-squareHalfSize, COMyx[1]+squareHalfSize+1]
    )[0]

tifffile.imwrite(f"{greyFile[0:-4]}-cropped.tif", greyOut)
