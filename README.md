![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)

# Single Cell Aligner
Developed by the [EPFL Center for Imaging](https://imaging.epfl.ch) for the [Living Patterns Laboratory](https://www.epfl.ch/labs/lpl/) in Sept 2023

We helped the LPL register... in order to...

## Alignment of high-freq images of a single cell image
Code to try to align images of a fast-moving cell with metachronal waves.
It uses the Digital Image Correlation approach in [spam](http://spam-project.gitlab.io/spam/), leveraging an easy-to-compute binary mask to only correlate the inside of the cell.

First make a mask (gaussian filter and threshold?) and use `02-maskCropping.py`

Then use `03-incrementalRegistration.py` to compute incremental registrations

<!--## Example result
<video width="320" height="240" controls>
  <source src="images/illustration.webm" type="video/webm">
</video>-->

## Authors
  - Edward Andò [EPFL Center for Imaging](https://imaging.epfl.ch)
  - Daphne Laan [EPFL Living Patterns Laboratory](https://www.epfl.ch/labs/lpl/)
  - Merih Ekin Özberk [EPFL Living Patterns Laboratory](https://www.epfl.ch/labs/lpl/)
