# Model-Based Iterative Reconstruction (MBIR) for Transmission Electron Microscopy (TEM)
#### Written by Aurys Šilinga

## Features:
1. Automated separation of electrostatic and magnetic components in TEM electron phase images. Uses feature matching from [fpd](https://gitlab.com/fpdpy/fpd) and [scikit-image](https://scikit-image.org/)
2. Alignment of TEM tomographic tilt series. Applicable to axially symmetric samples (cylinders, cones, needles).
3. 3D magnetisation reconstruction from phase image tilt series. Uses a modified version of [pyramid](https://iffgit.fz-juelich.de/empyre/empyre) for the calculation.
4. Example notebooks and video tutorials for all of the above.

## Installation:



## TODO:
* Format documentation.
* Implement CPU acceleration.
* Move main development into GPU accelerated branch.
* Replace all of pyramid back-end with more modern solutions.

## Development direction:
* 3D reconstruction of charge density.
* 3D mapping of material phases.
* GPU acceleration and distributed computing for larger datasets.
* Replace minimisation algorithm and reevaluate optimal estimation diagnostics.
