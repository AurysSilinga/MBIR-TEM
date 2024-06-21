# Model-based Iterative Reconstruction (MBIR) of 3D magnetisation for Transmission Electron Microscopy (TEM)
#### Written by Aurys Šilinga

## Features:
1. Automated separation of electrostatic and magnetic components in TEM electron phase images. Uses feature matching from [fpd](https://gitlab.com/fpdpy/fpd) and [scikit-image](https://scikit-image.org/)
2. Alignment of TEM tomographic tilt series. Applicable to axially symmetric samples (cylinders, cones, needles).
3. 3D magnetisation reconstruction from phase image tilt series. Uses a modified version of [pyramid](https://iffgit.fz-juelich.de/empyre/empyre) for the calculation.
4. Example notebooks and video tutorials for all of the above.

## Installation:
1. Set up a Python package manager. 

   Install Miniforge, which comes with a fresh installation of Python (for coding) and Mamba (for managing python packages).
   Follow the installation instructions on the [Miniforge website](https://conda-forge.org/miniforge).
   This was last tested with Miniforge 3-24.3.0-0 release for Windows x86_64.
2. Download MBIR-TEM source code.
3. Open the Miniforge Command Prompt (usually has a start menu shortcut) and navigate to the MBIR-TEM source code folder.

   On Windows navigate using
   
   `cd ".\download_folder_path\MBIR-TEM-main"`
4. In Miniforge Prompt run the command


   `mamba env create --file environment.yml`.
   
   This will download and install the necessary Python packages for MBIR-TEM to work.
5. Activate the MBIR-TEM environment

    `mamba activate mbir-tem`
6. Extract jutil-master.zip and navigate to the folder with the command 

   `cd jutil-master`

7. Run the command 

   `python setup.py install`
   
   This will install the Jutil package that is not available online for automatic download.
8. Return to the pyramidas-by-AS folder with the command 

   `cd ..`

9. Run the command 

   `python setup.py install`
   
    Now that all the prerequisite packages are installed, this will install MBIR-TEM itself.
10. Navigate to the tests folder with the command 

    `cd tests`
   
11. Run the command

    `python -m unittest`
   
    This will run all the tests and report if pyramidas was installed correctly. If something is wrong, an error will be shown.
    If the installation was successful, deprecation warnings are expected, but at the end, it should display 
    ```
     Ran X tests in Ys
     OK (skipped=6)
    ```


## TODO:
* Format documentation.
* Implement CPU acceleration.
* Move main development into GPU accelerated branch.
* Replace all of pyramid back-end with more modern solutions.
* Replace minimisation algorithm and reevaluate optimal estimation diagnostics.

## Development direction:
* 3D reconstruction of charge density.
* 3D mapping of material phases.
* GPU acceleration and distributed computing for larger datasets.
