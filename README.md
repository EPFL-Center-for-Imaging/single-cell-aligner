![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)

# Single Cell Aligner
Developed by the [EPFL Center for Imaging](https://imaging.epfl.ch) for the [Living Patterns Laboratory](https://www.epfl.ch/labs/lpl/) in Sept 2023

We helped Daphne Laan from the LPL to stabilise a series of beautiful high-speed (2000 FPS) microscopy images of the inside of a Didinium cell as it moves, allowing easier analysis of the waves going around the outside.

## Alignment of high-freq images of a single cell image
This repository contains python code to align images of a fast-moving cell with metachronal waves.

These don't seem to work in classical Fiji plugins such `stackreg`, so instead we use the Digital Image Correlation approach in [spam](http://spam-project.gitlab.io/spam/), the main points being:
  - It works for *very* large deformations
  - The correlation zone can simply be passed as a binary mask, allowing the algorithm to focus on what we want (*i.e.*, only the inside of the cell)

The alignment is performed as follows, starting from `originalStack.tif`, your 2D+t greyscale image sequence:
  1. make a binary mask (Suggestion: Large-scale Gaussian filter then threshold) to only catch the inside of the cell, and save as TIFF stack, e.g., `maskStack.tif`
  2. call `incrementalRegistration.py originalStack.tif maskStack.tif` which will:
    - do incremental registrations
    - sum them up correctly
    - apply them to each image
    - save `originalStack-registered-step20.tif` and a `did_spin1_1_1-PhisTotal-step20.npy` for debugging.

Useful parameters to change (directly in `incrementalRegistration.py`):
  - `step`: The number of frames to skip for incremental registrations, this should be set in such a way to have enough movement between frames in order not to have too much noise, and not to have too much movement so that the registration does not converge.


## Example result
Here you see the raw data (slowed down), a crop that centers the cell based on the centre of the mask, and finally the result of undoing the incremental registrations from `incrementalRegistration.py` which fully stabilises the cell... check out those metachronal waves!

![GIF of Didinium Cell](images/illustration.gif)
N.B. This GIF is very compressed and has a lot of frames dropped

## Authors
  - Edward Andò [EPFL Center for Imaging](https://imaging.epfl.ch)
  - Daphne Laan [EPFL Living Patterns Laboratory](https://www.epfl.ch/labs/lpl/)
  - Merih Ekin Özberk [EPFL Living Patterns Laboratory](https://www.epfl.ch/labs/lpl/)
