# Single Cell Aligner - EPFL Center for Imaging

![EPFL Center for Imaging logo](https://imaging.epfl.ch/analysis_projects/resources/assets/logo.svg)

## Alignment of high-freq images of a single cell image
Code to try to align images of a fast-moving cell with metachronal waves.
It uses the Digital Image Correlation approach in [spam](http://spam-project.gitlab.io/spam/), leveraging an easy-to-compute binary mask to only correlate the inside of the cell.

First make a mask (gaussian filter and threshold?) and use `02-maskCropping.py`

Then use `03-incrementalRegistration.py` to compute incremental registrations

## Authors
  - Edward Andò EPFL Center for Imaging
  - Daphne Laan EPFL Living Patterns Laboratory
  - Merih Ekin Özberk EPFL Living Patterns Laboratory
