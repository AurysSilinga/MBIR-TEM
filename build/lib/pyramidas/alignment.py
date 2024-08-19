# -*- coding: utf-8 -*-
# Aurys Silinga, 2024
#

"""
Image alignment functions for pyramid(by)AS
"""
# plotting and reading/writing files
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pyramid as pr
import hyperspy.api as hs

# image processing
from fpd.tem_tools import orb_trans, optimise_trans, apply_image_trans
import skimage.transform as sktr
from skimage.feature import canny
import skimage.filters as skfl

# fitting algorithms
import scipy.optimize as op
from scipy.fftpack import fftn, ifftn
from scipy.optimize import root_scalar, minimize
from scipy.optimize import root as root_multi
import itertools



from .util import *

#used in phase separation

def equalise_hspy_signals (s1_orig, s2_orig, fringe_spacing, plot_original = False, plot_cropped = True):
    """
    Takes two hyperspy signals, rebins them, and pads to have equal sizes.
    
    Returns
    (s1, s2) : tuple of hyperspy signals
        The rebinned and cropped images.
    """
    
    
    #inspect dimensions
    print("Original image side 1", s1_orig)
    print("Original image side 2", s2_orig)

    #Get pixel size from dm files and check scaling
    xaxis,yaxis=s1_orig.axes_manager.signal_axes
    a_spacing1=float(f"{xaxis.scale:.5g}")
    xaxis,yaxis=s2_orig.axes_manager.signal_axes
    a_spacing2=float(f"{xaxis.scale:.5g}")

    if a_spacing1 == a_spacing2:
        print("Original pixel spacing =",a_spacing2, xaxis.units)
    else:
        print("Original 1 pixel spacing =",a_spacing1, xaxis.units)
        print("Original 2 pixel spacing =",a_spacing2, xaxis.units)
        raise(Exception("Images have different magnifications"))

    #plot images
    if plot_original:
        s1_orig.plot("Side 1")
        s2_orig.plot("Side 2")
        
    print("\n*****\n")
    
    #rebin to match nyquist frequency
    nyquist_spacing = fringe_spacing/a_spacing1/2 #pixels
    rebin_factor = 2 ** (nyquist_spacing//2)
    
    #rebin
    print("Rebinning by", rebin_factor)
    s1r=s1_orig.rebin(scale=(rebin_factor,rebin_factor))/rebin_factor**2
    s2r=s2_orig.rebin(scale=(rebin_factor,rebin_factor))/rebin_factor**2
    
    #get pixel spacing
    xaxis,yaxis=s1r.axes_manager.signal_axes
    a_spacing=float(f"{xaxis.scale:.5g}")
    
    im1, im2 = pad_images_to_same_size(s1r.data, s2r.data)

    s1=hs.signals.Signal2D(im1)
    s2=hs.signals.Signal2D(im2)
    
    #propagate axis information
    for i in range(len(s1r.axes_manager.signal_axes)):
        s1.axes_manager.signal_axes[i].scale=s1r.axes_manager.signal_axes[i].scale
        s1.axes_manager.signal_axes[i].units=s1r.axes_manager.signal_axes[i].units
        
        s2.axes_manager.signal_axes[i].scale=s2r.axes_manager.signal_axes[i].scale
        s2.axes_manager.signal_axes[i].units=s2r.axes_manager.signal_axes[i].units
    
    print("Rebinned image side 1", s1)
    print("Rebinned image side 2", s2)
    print("Rebinned pixel spacing =",a_spacing, xaxis.units)

    #inspect cropped images
    if plot_cropped:
        s1.plot("Side 1")
        s2.plot("Side 2")

    return (s1, s2, a_spacing)


def fill_raw_dm_image(img, erode_width=1, radius=2, wrong_pixel_value=0):
    """
    Erodes the outer shell of pixels bordering free space, 
    and then fills in all free space by convolving pixels closest to edge.
    """
    wrongs = img==wrong_pixel_value
    #remove 1 pixel on image edges
    img, wrongs= erode_masked_image(img, wrongs, radius=erode_width, wrong_pixel_value=wrong_pixel_value) 
    #fill empty space with smooth extension
    img, wrongs = fill_masked_image(img, wrongs, radius=radius)
    return (img)


def identify_edges (im, imw, sigma = 4, high_threshold = 4, low_threshold = 0, roi=None):
    """
    Perform canny edge detection on the phase images so they could be measured for distortion later. 
    Once sample edge is identified, adjust sigma values to make orb transform give good alignment.
    
    sigma : float (optional)
        Gaussian blur applied to image before edge detection. 
        Value should be high enough to remove noise, but as low as possible to preserve sample edge shape
    high_threshold : float (optional)
        Filters such that only strongest edges are shown. Value should be high enough to only show the main sample edge.
    low_threshold: float (optional)
        Has no significant effect. See skimage.feature.canny documentation for more details.
        
    Returns
    (edge1,edge2) : tuple (N=2) of numpy arrays
        Images of sample edges.
    """
    # identify sample edges

    #     sigma=4.5 #
    #     low_threshold=0 #makes little difference
    #     high_threshold=4 #increase until only one edge remains

    edges1 = canny(im, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    edges2 = canny(imw, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    
    plot1=edges1.copy()
    plot2=edges2.copy()
    if roi is not None:
        wrongs=np.logical_not(roi) 
        plot1[wrongs] = 0
        plot2[wrongs] = 0
    
    matshow_n([plot1,plot2], labels=["side 1 edge","side 2 edge"])
    return (edges1.astype(float), edges2.astype(float))


def measure_orb_transform(edge1, edge2, image_warped = True, optimise = True, plot_result=True, 
                          verbose = True, save_trans=True, trans_path="transform.pickle", roi=None):
    """
    Measure the transformation between two images. Works best if initial images are already close or have distinct features.
    image_warped: bool (optional)
        If True, assumes that the image has shear and scaling is directional. otherwise transform euclidean. 
        If string is passed, applied 'translation', 'euclidean', 'similarity', or 'affine' transform
    optimise: bool (optinal)
        Perform sub-pixel optimisation to determine best transform.
        
    Returns
    model : skimage.transform._geometric 
        The optimised transform.
    """
    
    if image_warped == True:
        trans='affine' # has shear
    elif image_warped == False:
        trans='euclidean'  #only rotation and translation
    else:
        trans = image_warped
        
    fit1=edge1.copy()
    fit2=edge2.copy()
    if roi is not None:
        wrongs=np.logical_not(roi) 
        fit1[wrongs] = 0
        fit2[wrongs] = 0

    trans_meas = orb_trans(fit1, fit2, plot=True, residual_threshold=8, optimise=optimise, trans=trans)
    
    if verbose:
        try:
            print("\nTransformation")
            print("translation =",trans_meas.translation, "*( dx,dy)")
            print("roation =", np.degrees(trans_meas.rotation), "deg, clockwise")
            if image_warped:
                print("scale =",trans_meas.scale, "* (sx,sy)")
                print("shear", trans_meas.shear)
        except:
            print("Transformation values:\n", trans_meas)

    #apply transform to edges and inspect
    if plot_result:
        fit2_tr = apply_image_trans(fit2, trans_meas)
        #compare edges before and after transformation
        matshow_n([fit1 - fit2, fit1 - fit2_tr],["side_1 - side_2_uncorrected"," side_1 - side_2_transformed"])
    
    if save_trans:
        with open(trans_path, 'wb') as f:
            pickle.dump(trans_meas, f)
        print("Transformation saved as:",trans_path)
    return (trans_meas)
    

def get_space_around_mask(wrongs, starting_point=(0,0)):
    """
    Select free space around mask, but do not include holes contained within.
    """
    #surround by allowed space and borders
    wrongs = np.pad(wrongs, 1, constant_values=True)
    wrongs = np.pad(wrongs, 1, constant_values=False)
    
    yshape, xshape = wrongs.shape
    space_array=np.full((yshape,xshape), False)
    py,px = starting_point
    starting_point = (py+1,px+1)
    
    #run recursive search algorithm to fill all reachablespace
    span_fill(starting_point, space_array, wrongs, (0,yshape-1), (0,xshape-1))
    
    multi_slice = tuple([slice(2,-2)])*wrongs.ndim
    connected_space = space_array[multi_slice] #trim to original size
    
    return(connected_space)


def make_projection_data_simple(mag_phase, mask, confidence=None, zrot=0, xtilt=0, camera_rot=0, pix_spacing=1, subcount=5, dim=None, 
                                plot_results=True, save_data=False, data_path="data.pickle"):
    """
    Converts a single projection into a dataset containing the phase, projector, and metadata.
    """

    #dimensions
    if dim==None:
        dim=(1,*mag_phase.shape)
    dimz,dimy,dimx=dim   

    #initiate empty dataset
    data = pr.DataSet(pix_spacing, dim)

    x_ang=np.radians(xtilt)
    z_ang=np.radians(zrot)
    c_ang=np.radians(camera_rot)

    print("Reconstruction voxel number:", dimx*dimy*dimz)
    print("Pixel size: %.4f nm"%data.a)
    print("3d reconstructions dimensions:",data.dim)


    #create projector
    RP_Projector=pr.projector.RotTiltProjector
    print("starting projector calculation")
    projector = RP_Projector(dim, z_ang, x_ang, camera_rotation=c_ang, subcount=subcount, dim_uv=mag_phase.shape)
    print("projector calculation finished")

    #create phasemap object
    phasemap=pr.PhaseMap(pix_spacing, mag_phase, mask=mask, confidence=confidence)

    data.append(phasemap, projector)
    data.set_3d_mask()
    
    if plot_results:
        data.plot_phasemaps()

    if save_data:
        with open(data_path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(data, f)
            print("Data saved as:",data_path)
    
    return(data)


# used in tilt series alignment

def pad_equalise_tilt_series(phasemaps, make_square=True, pre_trim=True):
    """
    Takes an array of phasemaps, trims empty space, and then pads to make all the same size.
    Adjusts pixel spacing too.
    
    Returns: list, phasemaps_padded
        list of phasemaps that have the same shape and the same pixel spacing
    """
    if pre_trim: #remove empty space
        phasemaps=trim_empty_space(phasemaps)
    
    pixel_spacings=[]
    sizes=[]
    for pm in phasemaps:
        pixel_spacings.append(pm.a)
        sizes.append(pm.dim_uv)
        
    pixel_spacings = [pm.a for pm in phasemaps]
    spacing_average = np.average(pixel_spacings)
    spacing_average = float("%.5e"%spacing_average) #keep 5 significant figures
    spacing_std = np.std(pixel_spacings)
    print("Pixel spacing is %.3f +/- %.2e"%(spacing_average, spacing_std))
    
    sizes=np.array(sizes)
    x_max = np.max(sizes[:,1])
    y_max = np.max(sizes[:,0])
    
    if make_square:
        x_max = np.max((x_max, y_max))
        y_max = np.max((x_max, y_max))
    
    phasemaps_padded=[]
    for pm in phasemaps: #pad and adjust pixel spacing parameters
        dy,dx= pm.dim_uv
        pady=y_max - dy
        pady = (pady//2, pady-pady//2)
        padx = x_max - dx
        padx = (padx//2, padx-padx//2)
        
        pm_padded = pm.pad((pady,padx))
        pm_padded.a = spacing_average
        phasemaps_padded.append(pm_padded)
    return (phasemaps_padded)


def line_fn(x,a,b):
    return(a*x+b)


def fit_line(x, y, plot=True, fun=line_fn, p0=[1,1]):
    """
    Fits line, plots the fit, returns [fit,pop,pop_stderr]
    """
    pop,pcov=op.curve_fit(fun,x,y,p0=p0)
    fit=fun(x,*pop)
    if plot:
        plt.figure()
        
        plt.plot(x,y,'.',label='data')
        plt.plot(x,fit,label='fit')
        plt.legend()
        plt.show()
    return(fit,pop,np.sqrt(np.diag(pcov)))

def fit_mask(mask, crop_right=None, crop_left=None, crop_top=None, crop_bottom=None, 
             plot_results=True, x_tilt=None):
    """
    fits a line by minimising vertical distance of all pixels in given window.
    """
    #correct indexing
    if crop_top == 0:
        crop_top = None
    else:
        crop_top = -crop_top
        
    if crop_right == 0:
        crop_right = None
    else:
        crop_right = -crop_right
    
    mask_orig=mask.copy()
    mask=mask_orig[crop_bottom:crop_top, crop_left:crop_right]
    x_index=np.arange(mask.shape[1])
    y_index=np.arange(mask.shape[0])
    xv, yv = np.meshgrid(x_index, y_index)
    x_coord=xv[mask]+crop_left
    y_coord=yv[mask]+crop_bottom
    
    fit,pop,err=fit_line(x_coord, y_coord, plot=False)
    a,b=pop
    
    if plot_results:
        plt.figure()
        plt.imshow(mask_orig, origin='lower')
        plt.plot(x_coord, fit, 'r-')
        if x_tilt is None:
            plt.title(f"slope = {np.degrees(np.arctan(a)):.2f} deg")
        else:
            plt.title(f"x_tilt = {x_tilt:.2f} deg, slope = {np.degrees(np.arctan(a)):.2f} deg")
        xmin=np.min(x_coord) 
        xmax=np.max(x_coord)
        ymin=np.min(y_coord)
        ymax=np.max(y_coord)
        plt.plot([xmin,xmin,xmax,xmax,xmin],[ymin,ymax,ymax,ymin,ymin],'b-') #draw fitting region
        plt.show()
        
    return(a,b)


def measure_wire_orientations(phasemaps, projectors, crop_right=20, crop_left=20, crop_top=0, crop_bottom=0,
                              theta = 0, xtilt0 = 0, plot_results=True, verbose=True):
    """
    Fits lines to mask images, to determine alpha - the angle of the wire in each image.
    Also extracts the x-tilt from projectors.
    crop_right and crop_left should be set, such that the wire appears cylidrical.
    theta, xtilt in radians
    
    theta: float, default: 0
        angle (in radias) by which to rotate the images before fitting. 
        Used to make the rotation axis be horizontal. 
    xtilt0: float, default: 0
        tilt angle (in radias) at which the wire lies flat.
    
    returns lists [..., tan(alpha)_n, ...], [..., x_tilt_n, ...])
    """

    xtilts=[]
    tana=[]

    for i in range(len(phasemaps)):

        #get projection x-tilt
        xtilt=projectors[i].tilt + xtilt0
        xtilts.append(xtilt)

        #measure tan(alpha) of wire
        mask=phasemaps[i].mask
        mask=sktr.rotate(mask, np.degrees(theta) , resize=False) #rotate +ve direction is x->y
        mask=np.where(mask>0.1, True, False)
        a,b = fit_mask(mask, crop_right=crop_right, crop_left=crop_left, 
                       crop_top=crop_top, crop_bottom=crop_bottom,
                       plot_results=plot_results, x_tilt = np.degrees(xtilts[i]))
        tana.append(a)
        
        if verbose:
            print("x_tilt:",np.degrees(xtilts[i]),"deg")
            print("tan a:",a)
            print("cos x_tilt:",np.cos(xtilt))
            print()
            
    return(tana, xtilts)
    
    
def find_image_shifts (img_array, method="cross_correlation", **kwargs):
    """
    Legacy function.
    Assuming projections are similar, estimate their translations relative to each other.
    Calls either 'find_image_shifts_com' or 'find_image_shifts_correlation'.

    img_array: array of images 
        [img_1, img_2, ..., img_n]
    method: string 
        Either "cross_correlation" or "centre_of_mass"
        
    Returns: list of float tuples, 
        list with shifts for every image, of form [(sy_1,sx_1), .... , (sy_n, sx_n)]
    """
    if method=="cross_correlation":
        shifts=find_image_shifts_correlation(img_array, **kwargs)
    elif method=="centre_of_mass":
        shifts=find_image_shifts_com(img_array, **kwargs)
    else:
        raise ValueError("method not recongnised")
        
    return (shifts)

def find_image_shifts_com (img_array, test_image_index=None, com_to_centre=False, subpixel=False):
    """
    Measure relative centre of mass (CoM) translation of an array of images.
    
    img_array: array of images 
        [img_1, img_2, ..., img_n]
    test_image_index: int, default: middle index of img_array
        the central image that all others are aligned to match
    com_to_centre: bool, Default: False
        if True, centre of mass of each image is moved to the central coordinate regardless test_image position.
    subpixel: bool, Default:False
        if True, shifts should be rounded to closest pixel value

        
    Returns: list of float tuples, [(sy1,sx1), .... , (syN, sxN)]
        shifts between centres of mass relative to test_image CoM or to the centre of each image.
    """
    if test_image_index is None:
        test_image_index=len(img_array)//2
    
    shifts=[]
    test_image=img_array[test_image_index]
    dimy, dimx = test_image.shape
    for i,img in enumerate(img_array):
        x,y = np.arange(dimx), np.arange(dimy) 
        xv,yx=np.meshgrid(x, y)
        x_com = np.sum(xv*img)/np.sum(img)
        y_com = np.sum(yx*img)/np.sum(img)
        shifts.append(((dimy/2-0.5-y_com), (dimx/2-0.5-x_com)))
    if not com_to_centre: #rearange the shifts to be around a central position
        yc,xc=shifts[test_image_index]
        shifts=[(yx[0]-yc, yx[1]-xc) for yx in shifts]
    if not subpixel: # round the measurements
        shifts=[(round(yx[0]), round(yx[1])) for yx in shifts]
    return(shifts)

def find_image_shifts_correlation (img_array, upsample_factor=1):
    """
    Phase cross correlate img_n to img_n+1 and return an array of shifts relative to first image
    img_array: array of images 
        [img_1, img_2, ..., img_n]
    upsample_factor: int, default: 1
        Images will be registered to within 1 / upsample_factor of a pixel. 
    
    Returns: list of float tuples, 
        list with shifts for every image, of form [(sy_1,sx_1), .... , (sy_n, sx_n)]
    
    """
    shifts=[]
    shift_previous=np.array((0.0,0.0)) #starting shift vector
    for i,img in enumerate(img_array):
        if i==0: #compare first image with itself
            shifts.append(shift_previous.copy()) # for first image
            continue
        reference_image=img_array[i-1]
        img=img_array[i]
        shift,error,phasediff = phase_cross_correlation(reference_image, img, upsample_factor=upsample_factor)
        shift_previous+=shift #keep track of total shift
        shifts.append(shift_previous.copy())

    shifts=np.array(shifts)
    if upsample_factor==1:
        shifts=shifts.astype(int) #clean up unit types if measuring full pixel translations
        
    return(shifts)


def pad_translate_tilt_series(images, shifts, offset_x=0, offset_y=0, make_odd=False, make_square=False):
    """
    Translates all images according to 'shifts' relative to each other. 
    Pads all the images accordingly to implement the relative shifts.
    Amount of padding is minimised, but image centre corrdinate is not preserved.
    Also makes the images square and odd-sized to reduce errors when rotating and projecting.
    Works on both images and phasemaps.
    
    
    make_odd: True
        makes the image centre corespond to an integer coordinate. 
        prevents error with pyramid projector centre definitions.
    make_square: True
        makes the images square. More predictable behaviour when rotating.
    """
    sy_max, sx_max = np.max(shifts, axis=0)
    sy_min, sx_min = np.min(shifts, axis=0)
    
    sy_max=np.max(sy_max, 0) #translation is relative to centre
    sx_max=np.max(sx_max, 0)
    sy_min=np.min(sy_min, 0)
    sx_min=np.min(sx_min, 0)
    
    images_shifted=[]
    for i,pm in enumerate(images):
        dy, dx = shifts[i]
        dy+= offset_y
        dx+= offset_x
        pm=pm.copy()
                
        #shape corrections
        if make_square:
            if hasattr(pm, 'mask'):
                dimy, dimx=pm.mask.shape
            else: dimy, dimx=pm.shape
            dimy=sy_max+dimy-sy_min
            dimx=sx_max+dimx-sx_min
            
            if dimy > dimx:
                dim_diff = dimy-dimx
                sx_max = sx_max + dim_diff//2
                sx_min = sx_min - (dim_diff-dim_diff//2)
            elif dimx > dimy:
                dim_diff = dimx-dimy
                sy_max = sy_max + dim_diff//2
                sy_min = sy_min - (dim_diff-dim_diff//2)
        
        if make_odd:
            if hasattr(pm, 'mask'):
                dimy, dimx=pm.mask.shape
            else: dimy, dimx=pm.shape
            dimy=sy_max+dimy-sy_min
            dimx=sx_max+dimx-sx_min
            
            if dimy%2 == 0:
                sy_min -= 1
            if dimx%2 == 0:
                sx_min -= 1
        
        pad_y=(-sy_min+dy, sy_max-dy)
        pad_x=(-sx_min+dx, sx_max-dx)
        if hasattr(pm, 'pad'):
            pm = pm.pad((pad_y,pad_x)) #if a phase map
        else:
            pm = np.pad(pm, (pad_y,pad_x)) #if a matrix
        images_shifted.append(pm)
        
    return(images_shifted)
    
    
def build_error_fn(tana1,tana2,xtilt1,xtilt2,theta=None,xtilt=None):
    """
    Returns a pointer to an error function that uses the given tan(a)(n), and xtilt(n) values.
    If values for theta or xtilt_0 are given, then that value is not used as a variable.
    Used to create error functions for a minimisation solver
    
    In radians
    """
    
    def error_fn_theta_pi(x, absolute):
        """
        Based on measurememnts of projection angle and wire orientation, calculates a scalar function 
        that is equal to 0 if correct values of miss-tilt (xtilt_0) and rotation axis orientation (theta) are chosen.
        
        Returns error on function value. If x = (th, xtilt_0) are correct solutions, returns 0.
        """
        
        if theta is None and xtilt is None:
            th,xtilt_0=x
        elif theta is None:
            th=x
            xtilt_0=xtilt
        elif xtilt is None:
            th=theta
            xtilt_0=x

        t1=tana1
        t2=tana2
        cp1=np.cos(xtilt1+xtilt_0)
        cp2=np.cos(xtilt2+xtilt_0)
        sth=np.sin(th)
        cth=np.cos(th)

        error=(sth*cth*(cp2*t1*t2-cp1*t1*t2-cp2+cp1) + sth*sth*(cp1*t1-cp2*t2) + cth*cth*(cp2*t1-cp1*t2))
        
        if absolute:
            error=np.abs(error) **2 #chi-square
        
        return(error)
    
    return (error_fn_theta_pi)


def fix_bracket(bracket, fun, args):

    fa=fun(bracket[0], *args)
    fb=fun(bracket[1], *args)
    if fa*fb >0:
        x=np.linspace(*bracket, num=100)
        y=fun(x, *args)
        bracket=[x[np.argmin(y)],x[np.argmax(y)]]

        fa=fun(bracket[0], *args)
        fb=fun(bracket[1], *args) 

    return bracket
    
    
def get_total_err_fn (tana, xtilts, absolute=True):
    i_s = range(len(xtilts))
    num = 2
    combinations = list(itertools.combinations(i_s, num))

    err_fns=[]

    for comb in combinations:
        i,j = comb
        tana1=tana[i]
        tana2=tana[j]
        xtilt1=xtilts[i]
        xtilt2=xtilts[j]
        err_fns.append(build_error_fn(tana1,tana2,xtilt1,xtilt2))
    
    def total_err_fn(x):
        ret=0
        for fn in err_fns:
            err = fn(x, absolute = absolute)
            ret = ret + err
        return (ret)
    
    return (total_err_fn)
    
def minimise_total_error(tana, xtilts, get_total_err_fn = get_total_err_fn, x0=[0,0], 
                         bounds=[[-np.pi/2, +np.pi/2], [-np.pi/2, +np.pi/2]], tolerance=0, method='TNC', 
                         options={'maxiter':300}, jac='cs', verbose=False, 
                         error_grid_num=100, error_window_width=np.radians(3)):
    """
    Takes a set of measured needle directions and tilts. Outputs the direction of the axis of rotation and the true starting tilt.
    For a volume of revolution (e.g. cylinders and cones), if the main axis of the needle is not parallel to the axis of rotation, 
    then the angle between these two axes can be predicted for any x-tilt. (The projections are trivially calculated using trigonometry)
    This symmetry is used to find the axis or tilting and the tilt angles for all phasemaps.
    
    tana: list
        list containing the tangents between symmetry axis and image x-axis for every image.
    xtilts: list
        tilt angles for each image. Same as shown on microscope during acquisition
    get_total_err_fn: function, optional
        function that returns the total error to be minimised
    x0: 2-tuple, default: (0,0)
        starting guess for tilt-axis direction and miss-tilt. In radians. 
        x-tilt is considered 0 when the needle lies flat in the image.
        miss-tilt measures angle between starting and flattest possible position
    bounds: optional
        allowed range of possible values. Default allows all possible axis orientations to be considered (-90 to +90 deg).
    tolerance: float, default: 0
        What absolute error is acceptable by the fitting algorithm.
    method: string, default 'TNC'
        Method to be used by scipy.optimize.minimise. Can use 'L-BFGS-B' or 'TNC'.
    options: optional
        See scipy.optimize.minimise.
    jac
        See scipy.optimize.minimise.
    error_grid_num: int, default: 100
        How many samples to evaluate inside 'error_window_width' when estimating errors
    error_window_width: float, default: 0.052
        distance from-centre-to-edge (in radians) when evaluating the error function for error estimation. Default is 3 degrees.
    
    """
    total_err_fn = get_total_err_fn (tana, xtilts)

    sol=minimize(total_err_fn, x0, bounds=bounds, tol=tolerance, method=method, options=options)
    if verbose:
        print(sol)
    args=sol.x
    args_deg=np.degrees(args)
    
    #error estimate by Gaussian fitting
    th0, p0 = args
    bracket = [[-error_window_width+th0, error_window_width+th0],[-error_window_width+p0, error_window_width+p0]]
    xv = np.linspace(*bracket[0], num=error_grid_num)
    yv = np.linspace(*bracket[1], num=error_grid_num)
    x,y = np.meshgrid(xv, yv) # grid of points
    data = total_err_fn([x, y]) # evaluation of the function on the grid
    try:
        popt, pcov = fit_2D_gaussian_blob((x,y), data, is_error_function=True, plot_results=True, labels=["miss-tilt, rad", "tilt-axis direction, rad"])
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt
        theta_err = [sigma_x]
        xtilt_err = [sigma_y]
        
        if (np.abs(th0-x0) > np.radians(1)) or (np.abs(p0-y0) > np.radians(1)):
            raise ValueError("Gaussian fitting is not centered on the solution!")
        
    except Exception as EXC:
        print(EXC)
        print("Could not fit a gaussian function to the data! Trying a grid search error estimation")
        #error estimate by grid search
        possible_sol = data < 1*sol.fun #~ 1 std
        if len(possible_sol[possible_sol]) > 0:
            theta_err = np.degrees(x[possible_sol]) - args_deg[0]
            xtilt_err = np.degrees(y[possible_sol]) - args_deg[1]
        else:
            theta_err = [0]
            xtilt_err = [0]
    
    
    print("root theta: %.2f deg +%.3f / -%.3f"%(args_deg[0],np.max(theta_err),np.abs(np.min(theta_err))))
    print("root x_tilt: %.2f deg +%.3f / -%.3f"%(args_deg[1],np.max(xtilt_err),np.abs(np.min(xtilt_err))))
    print("distance from 0 error:", sol.fun)
    
    return (sol, total_err_fn)


def plot_3D_surface(fun, args=[], bracket = [[-np.pi/2,+np.pi/2], [-np.pi/2,+np.pi/2]], xlabel="Theta", 
                    ylabel="Miss-tilt", title="Error fuction", num=75):
    x = np.linspace(*bracket[0], num=num)
    y = np.linspace(*bracket[1], num=num)
    
    X,Y = np.meshgrid(x, y) # grid of point
    Z = fun([X, Y], *args) # evaluation of the function on the grid

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z)
    plt.xlabel("Theta")
    plt.ylabel("Miss-tilt")
    plt.title(title)
    plt.show()
    

def rotate_phasemaps(phasemaps, axis_rot, resize=True, mask_threshold=0.5, smooth_masks=False, sigma=2,
                     mode='edge', make_odd=True):
    """
    Takes phasemaps. Rotates them by angle axis_rot (in degrees).
    
    """
    phasemaps_rot=[]
    
    #pad all
    for pm in phasemaps:
        
        #remove 0s to preserve values during rotation
        phase, temp_wrongs = fill_masked_image(pm.phase, pm.confidence<1)
        
        
        #rotate
        phase_rot = sktr.rotate(phase, axis_rot, resize=resize, mode=mode)
        
        conf_rot = sktr.rotate(pm.confidence, axis_rot, resize=resize, mode='constant')
        conf_rot = np.where(conf_rot > mask_threshold, 1, 0)
        
        mask_rot = sktr.rotate(pm.mask, axis_rot, resize=resize, mode='constant')
        if smooth_masks:
            mask_rot = skfl.gaussian(mask_rot, sigma=sigma)
        mask_rot=np.where(mask_rot>mask_threshold, True, False)
        
        pm=pr.PhaseMap(pm.a, phase_rot, mask_rot, conf_rot)
        
        #correct dimensions to preserve image odd-ness
        if make_odd:
            dimy,dimx=pm.dim_uv
            padding=[[0,0],[0,0]]
            if dimy%2 == 0:
                padding[0][0]=1
            if dimx%2 == 0:
                padding[1][0]=1
            pm=pm.pad(padding)
                
        phasemaps_rot.append(pm)
        
    return(phasemaps_rot)



def shift_phasemaps(phasemaps, shifts, padded=False):
    """
    shifts all phasemap[i] by (dy,dx)=shifts[i] and filling empty space with 0.
    padded=True to pad all phasemaps equally, such that no data is lost. 
    Padding preserves centre and pads all phasemaps the same.
    """
    
    phasemaps_shifted=[]
    if padded: #pad preserving centre
        pval = np.max(np.abs(shifts))
        phasemaps = [pm.pad(((pval,pval),(pval,pval))) for pm in phasemaps]
    
    for i,pm in enumerate(phasemaps): #shift all phasemaps
        phasemaps_shifted.append(pm.shift(shifts[i])) 
    return(phasemaps_shifted)

def find_edges(wrongs, axis=0):
    """
    finds the low and high index of the first slice that contains a correct pixel along an 'axis'
    """
    low_edge=None
    high_edge=None
    full=np.where(wrongs, 0, 1)
    
    #reduce the matrix into a single line along the axis in which we need to find first and last filled coordinate
    axes=list(range(full.ndim))
    axes.remove(axis)
    full_flattened=np.sum(full, axis=tuple(axes))
    
    for i in range(len(full_flattened)):
        if full_flattened[i]>0:
            low_edge=i
            break
    for i in range(len(full_flattened))[::-1]:
        if full_flattened[i]>0:
            high_edge=i
            break

    return (low_edge, high_edge)

def centre_phasemaps(phasemaps, padded=False, verbose=False):
    """
    Measures if the true phase image is in the centre of each phasemap and shifts toward the centre.
    """
    shifts=[]
    phasemaps_centered=[]
    
    for pm in phasemaps:
        wrongs=pm.confidence<1
        left,right=find_edges(wrongs, 1)
        bottom,top=find_edges(wrongs, 0)
        dy,dx = pm.dim_uv
        
        shift_x= ((dx-right-1) - left)//2
        shift_y= ((dy-top-1) - bottom)//2
        shifts.append((shift_y,shift_x))
    
    phasemaps_centered=shift_phasemaps(phasemaps, shifts, padded=padded)
    if verbose:
        print("centering shifts (index, (dy,dx)):", list(zip(range(len(shifts)),shifts)))
            
    return(phasemaps_centered)
    
    

def align_wire_tips(phasemaps, axis=1, use_high_end = True, padded=False, verbose=False):
    """
    Shift mask images to the left, such that the wire tips are 
    placed at the same distance from the right edge of the image.
    default axis=1 is x-axis.
    """

    from_edge=[]
    phasemaps_tr=[]
    
    #for each image, measure distance from tip to image edge
    for pm in phasemaps:
        free_space=np.bitwise_not(pm.mask)
        low_edge, high_edge = find_edges(free_space, axis=axis)
        if use_high_end:
            dim = pm.dim_uv[axis]
            from_edge.append(dim - high_edge - 1)
        else:
            from_edge.append(low_edge)
            
    from_edge=np.array(from_edge)
    if axis==1:
        shifts=from_edge - (np.max(from_edge) + np.min(from_edge))//2
        if not use_high_end:
            shifts=-1*shifts
    else: #estimate x shift from y shift position of wire tip. Find x-axis COM at the tip y-coordinate  
        if use_high_end:
            raise NotImplementedError("Vertical tip finding at the top of the image is NOT IMPLEMENTED")
        else: #bottom of image with xCOM finding
            mask_rows=[]
            for i, pm in enumerate(phasemaps):
                row_i=from_edge[i]
                mask = pm.mask
                mask_row = mask[row_i:row_i+1,:]
                mask_row = np.where(mask_row, 1, 0)
                mask_rows.append(mask_row)
            COM_shifts=find_image_shifts(mask_rows, method="centre_of_mass")
            shifts=[shift[1] for shift in COM_shifts] #retain only x
        
    shifts_yx=[]
    for shift in shifts:
        el=(0,shift) #in numpy the order is [dy,dx].
        shifts_yx.append(el)
    
    phasemaps_shifted=shift_phasemaps(phasemaps, shifts_yx, padded=padded)
    if verbose:
        print("shifts (index, (dy,dx)):", list(zip(range(len(shifts_yx)),shifts_yx)))
        
    #cleanup:
    for pm in phasemaps_shifted:
        pm.confidence = np.where(pm.confidence<0.5, 0, 1)
        
    return(phasemaps_shifted)

def trim_empty_space(phasemaps, equal_trim = False, verbose=False):
    """
    crops all phasemaps to have minimum empty space, but preserves image centre.
    equal_trim = True if all phasemaps are even squares and should become smaller even squares.
    """
    
    #find the edges in all directions
    slices_indexes=[]
    for pm in phasemaps:
        wrongs = pm.confidence<1
        phasemap_edges=[]
        for axis in range(wrongs.ndim):
            low_i, high_i = find_edges(wrongs, axis=axis)
            high_i = high_i+1 #cut away pixel before the edge
            phasemap_edges.append((low_i, high_i))
        if verbose:
            print("[(low y, hight y), (low x, high x)]:", (phasemap_edges))
        slices_indexes.append(phasemap_edges)
        
    if equal_trim:
        min_i = np.min(slices_indexes)
        max_i = np.max(slices_indexes)
        
        dims = [pm.dim_uv for pm in phasemaps]
        if np.max(dims) != np.min(dims):
            raise ValueError ("Phasemaps are not all equal and square!")
        
        max_i_dist = np.max(dims) - max_i
        dist=np.min((min_i, max_i_dist))
        min_i = dist
        max_i = np.max(dims) - dist
        
        phasemaps_trimmed = [pm[(slice(min_i,max_i),slice(min_i,max_i))] for pm in phasemaps]
    else:
        phasemaps_trimmed=[]
        for i, indexes in enumerate(slices_indexes):
            multi_slice=[]
            for axis, axis_edges in enumerate(indexes):
                axis_length = phasemaps[i].dim_uv[axis]
                dist = np.min((axis_edges[0], axis_length - axis_edges[1])) #trip both sides equally to preserve centre
                multi_slice.append((slice(dist, axis_length-dist)))
            multi_slice=tuple(multi_slice)
            phasemaps_trimmed.append(phasemaps[i][multi_slice])
        
    return phasemaps_trimmed
    
    
def project_mask(mask0, projector, reproject=False, mask_threshold=0.5):
    """
    takes a 3-dimensional mask and a projector. 
    Applies the projector to calculate the projection.
    If the mask is a 2-d image then it is assumed to have a z-height of 1 pixel, and vice-versa.
    If reproject=True, takes a 2-d image and applies the inverse of a projection to calculate the equivalent 3-d mask.
    """
    
    #inverse projection
    if reproject:
        mask_2d = mask0.reshape(-1)  # vectorise
        mask_2d_projected=projector.weight.T.dot(mask_2d).reshape(projector.dim) #project and reshape
        if mask_2d_projected.shape[0] == 1:
            mask00=mask_2d_projected[0,...] #make into image
        else:
            mask00 = mask_2d_projected.copy()
        mask00=np.where(mask00>mask_threshold, True, False)  #correction for space-streching by z-rotation
        return (mask00)
    
    #projection
    else:
        if mask0.ndim == 2:
            mask_3d=np.zeros((1,*mask0.shape))
            mask_3d[0,:,:]=mask0
        else:
            mask_3d = mask0.copy()

        mask_field=pr.ScalarData(1,mask_3d)
        field_proj=projector(mask_field)
        mask01=field_proj.field[0,...]
        mask01=np.where(mask01>mask_threshold, True, False)
        return(mask01)    
    
def mask_to_3d_round(mask2d, axis=1, trim_z=True):
    """
    Takes a 2d mask and calculates a 3-d representation by assuming that each slice along an axis is projection of a disk.
    """
    #if estimating not along the x-axis, rotate the mask before calculation
    mask2d = np.rot90(mask2d, k=(axis+1)%2, axes=(0,1)) 
    dimy, dimx = mask2d.shape
    dimz = np.max((dimy,dimx))
    if dimz%2 == 0: #2d mask must lie in the centre
        dimz+=1
    mask_3d = np.full((dimz, dimy, dimx), False)
    
    for i in range(dimx):
        line = mask2d[:, i]
        start_j=None
        end_j=None
        
        for j, mask_pixel in enumerate(line):
            if (start_j is None) and (mask_pixel == True): #find first pixel of the edge
                start_j = j
            elif (start_j is not None) and (mask_pixel == False): #if edge is found
                end_j = j-1
            elif (start_j is not None) and (end_j is None) and (j == dimy-1): #final pixel is the edge
                end_j = dimy-1
                
            if (start_j is not None) and (end_j is not None):
                centre_j = (end_j + start_j)/2
                dist = (end_j - start_j)/2
                
                #find valid pixels in slice
                y = np.arange(dimy)
                z = np.arange(dimz)
                yv, zv = np.meshgrid(y, z)
                
                yv_centered = yv - centre_j
                zv_centered = zv - (dimz-1)/2
                distance_grid = np.sqrt(yv_centered**2 + zv_centered**2)
                valid_pixels=distance_grid <= dist
                mask_3d[valid_pixels,i] = True
                start_j=None
                end_j=None
        
    #if estimating not along the x-axis, unrotate the mask after calculation
    mask_3d = np.rot90(mask_3d, k=-(axis+1)%2, axes=(1,2)) 
    
    #remove empty space
    if trim_z:
        wrongs=np.bitwise_not(mask_3d)
        low_i, high_i = find_edges(wrongs, axis=0)
        high_i = high_i+1 #cut away pixel before the edge
        axis_length=wrongs.shape[0]
        dist = np.min((low_i, axis_length - high_i)) #trip both sides equally to preserve centre
        multi_slice=((slice(dist, axis_length-dist),slice(None),slice(None)))
        mask_3d=mask_3d[multi_slice]
        
    return(mask_3d)
    
def align_wire_directions(phasemaps, tilts, plot_fits=False, plot_aligned_masks=True, crop_right=40,
                          crop_left=40, crop_top=1, crop_bottom=0, test_mask_index=None, use_round_projection=True, 
                          axis=1, z_ang=0, camera_rotation=0, subcount=5, padded=False, verbose=False):
                          
    """
    
    if use round projection, axis defines whether y or x should be used as the symmetry axis
    """

    tilts=np.radians(tilts)
    z_ang = np.radians(z_ang)
    camera_rotation=np.radians(camera_rotation)
    RP_Projector=pr.projector.RotTiltProjector
    
    phasemaps_aligned=[]
    
    
    #start with finding flattest phasemap and forming a 3d model assuming needle is round
    if test_mask_index is None:
        test_mask_index = np.argmin(np.abs(tilts))
    
    #take the test mask and reproject to 0 deg tilt
    mask0=phasemaps[test_mask_index].mask 
    dim0=(1,*mask0.shape)
    x_ang=tilts[test_mask_index]
    projector0 = RP_Projector(dim0, z_ang, x_ang, camera_rotation=camera_rotation, subcount=subcount, dim_uv=mask0.shape)
    mask00=project_mask(mask0, projector0, reproject=True) #mask at true 0 tilt
    
    #fit line to the selected part of the wire
    param0 = fit_mask(mask00, plot_results=plot_aligned_masks, crop_right=crop_right, 
                      crop_left=crop_left, crop_top=crop_top, crop_bottom=crop_bottom, x_tilt=0)
    x_mask_mid = (mask00.shape[1]-1)/2
    y_mid_0 = line_fn(x_mask_mid,*param0)
    a0 = param0[0]

    print("0 tilt mask is calculated from mask",test_mask_index)
    
    
    #compare the projections to the experimental masks at various angles
    projectors=[]
    if use_round_projection:
        test_mask_3d = mask_to_3d_round(mask00, axis=axis)
        dim=test_mask_3d.shape
        for tilt in tilts:
            print("Starting projector calculation for tilt %.2f deg"%np.degrees(tilt))
            projector = RP_Projector(dim, z_ang, tilt, camera_rotation=camera_rotation, subcount=subcount, dim_uv=mask00.shape)
            projectors.append(projector)
    
    else:
        test_mask_3d=mask00 # Use 2d mask as if it is a flat surface in 3d
        dim=(1,*mask0.shape)
        for tilt in tilts:
            print("Starting projector calculation for tilt %.2f deg"%np.degrees(tilt))
            projector = RP_Projector(dim, z_ang, tilt, camera_rotation=camera_rotation, subcount=subcount, dim_uv=mask00.shape)
            projectors.append(projector)

        print("projector calculations finished")
        print()

    y_shifts=[]
    direction_errors=[]
    for i in range(len(projectors)):

        #project to certain angle
        projector=projectors[i]
        mask1=project_mask(test_mask_3d, projector)
        mask2=phasemaps[i].mask


        #find wire slopes
        param1=fit_mask(mask1, plot_results=plot_fits, crop_right=crop_right, crop_left=crop_left, crop_top=crop_top, crop_bottom=crop_bottom, x_tilt=np.degrees(projector.tilt))
        param2=fit_mask(mask2, plot_results=plot_fits, crop_right=crop_right, crop_left=crop_left, crop_top=crop_top, crop_bottom=crop_bottom, x_tilt=np.degrees(projector.tilt))

        a1,a2=param1[0],param2[0]
        alphas=(np.arctan(a1),np.arctan(a2))
        alphas_deg= np.degrees(alphas)
        direction_errors.append(alphas_deg[0]-alphas_deg[1])
        
        #find y shift values (x shifts were already fixed by aligning wire tips)
        y1=line_fn(x_mask_mid,*param1)
        y2=line_fn(x_mask_mid,*param2)
        dy=round(y1-y2)
        if verbose:
            print("Mask %d Y-shift = %d, Wire direction error = %.2f deg"%(i,dy,alphas_deg[1]-alphas_deg[0]))

        
        y_shifts.append(dy)
        
        if plot_aligned_masks:
            plt.figure()
            plt.subplot(121)
            plt.imshow(mask1, origin='lower')
            plt.title("test mask projected at %.2f deg tilt"%(np.degrees(projector.tilt)))

            plt.subplot(122)
            pm = phasemaps[i]
            pm = pm.shift((dy,0))
            plt.imshow(pm.mask, origin='lower')
            plt.title("mask %d corrected at %.2f deg tilt"%(i, np.degrees(projector.tilt)))
            plt.show()
            
    shifts_yx=[(dy,0) for dy in y_shifts]
    phasemaps_aligned=shift_phasemaps(phasemaps, shifts_yx, padded=padded)
    
    #correct finite precision errors
    for pm in phasemaps_aligned:
        pm.confidence = np.where(pm.confidence<1, 0, 1)
        pm.phase[pm.confidence<1]=0
    
    dir_errs=np.array(direction_errors)
    dir_errs = dir_errs*dir_errs
    dir_err = np.sum(dir_errs)/dir_errs.shape[0]
    dir_err = np.sqrt(dir_err)
    print(f"Average wire direction alignment error: {dir_err:.3f} deg")
    
    pm=pr.PhaseMap(1, np.zeros_like(mask00), mask=mask00, confidence = np.where(mask00, 1, 0))
    pm = trim_empty_space([pm], equal_trim=False, verbose=False)[0]
    mask_3d_trimmed = mask_to_3d_round(pm.mask)
    reconstruction_dimensions=mask_3d_trimmed.shape
    return(phasemaps_aligned, reconstruction_dimensions)
    
    
def make_projection_data(phase_maps, zrots, xtilts, camera_rots, pixel_spacing, center = None, subcount=5, dim=None, 
                                plot_results=True, save_data=False, data_path="data.pickle"):
    """
    add phasemaps into a pr.DataSet object.
    calculates the projectors for each phasemap as well
    """
    
    if dim==None:
        dim=[phase_maps[0].mask.shape[0]]*3
    dimz,dimy,dimx=dim   
    
    if not isinstance(center, (list, tuple)): #allow passing custom centres for each projection, or a single 3-tuple value, or None.
        center = [center]*len(phase_maps)
    elif not isinstance(center[0], (list, tuple)):
        center = [center]*len(phase_maps)
    
    #initiate empty dataset
    data = pr.DataSet(pixel_spacing, dim)

    x_angs=np.radians(xtilts)
    z_angs=np.radians(zrots)
    c_angs=np.radians(camera_rots)

    print("Reconstruction voxel number:", dimx*dimy*dimz)
    print("Pixel size: %.4f nm"%data.a)
    print("3d reconstructions dimensions:",data.dim)


    #create projector
    RP_Projector=pr.projector.RotTiltProjector
    
    print("starting projector calculation")
    for i in range(len(phase_maps)):
        
        #create phasemap object
        phasemap=phase_maps[i]
        
        x_ang=x_angs[i]
        z_ang=z_angs[i]
        c_ang=c_angs[i]
        
        projector = RP_Projector(dim, z_ang, x_ang, camera_rotation=c_ang, center=center[i], subcount=subcount, dim_uv=phasemap.dim_uv)
        print("%d/%d"%(i+1,len(phase_maps)),end="; ")

        data.append(phasemap, projector)
    print("projector calculation finished\n")
        
    data.set_3d_mask()
    data.set_Se_inv_diag_with_conf()
    
    if plot_results:
        data.plot_phasemaps()

    if save_data:
        with open(data_path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(data, f)
            print("Data saved as:",data_path)
    
    return(data)


def save_editable_mask(mask,axis=2, label="mask_projection"):
    """
    save a projection of the mask(3D boolead array) as a png that can be easily edited
    returns filename
    """
    if len(mask.shape) > 2:
        mask_sum=np.sum(np.where(mask,1,0),axis=axis)
    elif len(mask.shape)==2:
        mask_sum=mask.copy()
    mask_sum=mask_sum*255//np.max(mask_sum)
    plt.matshow(mask_sum)
    plt.title("mask projection")

    mask_slice_rgb=np.zeros((*mask_sum.shape,4), dtype='uint8') #make empty .png
    mask_slice_rgb[:,:,-1]=255 #set intensity
    mask_slice_rgb[:,:,0]=mask_sum #set colour value
    print("png shape", mask_slice_rgb.shape)

    file_type=".png"
    img=Image.fromarray(mask_slice_rgb)
    img.save(label+file_type)
    
    return(label+file_type)
    
    
def load_png_mask(mask, filename, axis=2):
    """
    loads an edited .png projection of a mask and edits the the given mask to match.
    returns updated mask
    """
    img=Image.open(filename)
    mask_x_paint=np.array(img)
    print(".png shape",mask_x_paint.shape) #shape is (r,g,b,a)
    mask_x_fix=mask_x_paint[:,:,:-1].sum(axis=2) #anything not black is included in the mask.
    mask_x_fix=mask_x_fix.astype(bool)
    
    if mask is None:
        mask = mask_x_fix
    else:
        x_shape=mask.shape[axis]
        mask_fix=np.tile(mask_x_fix, (x_shape,1,1)) #stack as a pile along z axis
        mask_fix=np.rollaxis(mask_fix, 0,axis+1) #roll z axis to be along x
        print("mask shape",mask_fix.shape)
        plt.matshow(np.sum(mask_fix, axis=axis))
        plt.title("sum of mask fix")

        mask=np.logical_and(mask, mask_fix)
    return(mask)

def rotate_3d_mask(data_series, plot_results=True):
    """
    Acts in place.
    Rotates the 3d mask by 90 degrees and takes the overlap to impose symmetry
    """
    mask0 = data_series.mask
    mask90 = np.rot90(mask0, axes = [0,1])
    mask_both = np.logical_and(mask0, mask90)
    data_series.mask=mask_both
    if plot_results:
        data_series.plot_mask(title = "Rotated mask")
        matshow_n([np.sum(data_series.mask, axis=0), np.sum(data_series.mask, axis=1)],
          ["z projection after rotation","y projection after rotation"])


def project_scalar_array(array, zrot=0, xrot=0, crot=0, dim_uv=None, subcount=1, center=None, center_offset=(0,0,0)):
    scalar_field = pr.ScalarData(1, array.copy())
    RP_Projector=pr.projector.RotTiltProjector
    dim=array.shape
    dim_z, dim_y, dim_x = dim
    if center == None:
        center = (dim_z / 2.+center_offset[0], dim_y / 2.+center_offset[1], dim_x / 2.+center_offset[2])
    if dim_uv is None:
        dim_uv=(np.max(array.shape),np.max(array.shape))
    projector = RP_Projector(dim, np.radians(zrot), np.radians(xrot), camera_rotation=np.radians(crot), 
                             subcount=subcount, dim_uv=dim_uv, center=center)
    field_proj = projector(scalar_field)
    proj=field_proj.field[0,...]
    return(proj)

def pad_img_to_square(img, mode='constant'):
    """
    pad images to be square
    preserves the centre
        
    return: padded_img, ndarray
    """

    y1,x1=img.shape
    pad=np.abs(y1-x1)
    pad1=pad//2
    pad2=pad//2
    
    if pad%2 == 1:
        pad2=pad2+1
    
    if y1==x1:
            pass
    elif y1>x1:
        img=np.pad(img,((0,0),(pad1,pad2)), mode=mode)
    else:
        img=p.pad(img,((pad1,pad2),(0,0)), mode=mode)
    
    return(img)




































# Possibly unused

def smooth_masked_img(img, sigma=10, conf=None, recrop=False):
    """
    smooths and rebins a hyperspy signal
    
    Returns
    signal : hyperspy signal
        
    """
    if conf is None:
        conf=np.ones(img.shape)
    
    #handle edges for non-rectangular images
    data=fill_img_mask(img, conf)

    #gaussian filter
    data=gfilter(data, sigma=sigma)
    
    #remove false parts of image
    if recrop:
        data[conf==0]=0
    
    return(data)




        
def equalise_dm (s1_orig, s2_orig, rebin_factor, side_2_rotation = 0, plot_original = False, plot_cropped = True):
    """
    Takes two hyperspy signals, rebins them, and pads to have equal sizes.
    
    Returns
    (s1, s2) : tuple of hyperspy signals
        The rebinned and cropped images.
    """
    
    
    #inspect dimensions
    print("Original image side 1", s1_orig)
    print("Original image side 2", s2_orig)

    #check scales
    xaxis,yaxis=s1_orig.axes_manager.signal_axes
    a_spacing1=xaxis.scale
    xaxis,yaxis=s2_orig.axes_manager.signal_axes
    a_spacing2=xaxis.scale

    if a_spacing1 == a_spacing2:
        print("Original pixel spacing =",a_spacing2, xaxis.units)
    else:
        print("Original 1 pixel spacing =",a_spacing1, xaxis.units)
        print("Original 2 pixel spacing =",a_spacing2, xaxis.units)
        raise(Exception("Images have different magnifications"))

    #plot images
    if plot_original:
        s1_orig.plot()
        s2_orig.plot()
        
    print("\n*****\n")
    #rebin
    s1r=s1_orig.rebin(scale=(rebin_factor,rebin_factor))/rebin_factor**2
    s2r=s2_orig.rebin(scale=(rebin_factor,rebin_factor))/rebin_factor**2

    #rotate image 2 if necessary
    im2=sktr.rotate(s2r.data, side_2_rotation, mode='constant', resize=True)
    
    #pad images to have same size
    dy2,dx2=im2.shape
    dy1,dx1=s1r.data.shape
    
    pady=dy2-dy1
    padtop=pady//2
    padbottom=pady//2
    if pady%2 == 1:
        padbottom=padbottom+1
        
    padx=dx2-dx1
    padleft=padx//2
    padright=padx//2
    if padx%2 == 1:
        padright=padright+1
    
    im1 = np.pad(s1r.data, ((padtop,padbottom),(padleft,padright)), mode='constant' )

    s1=hs.signals.Signal2D(im1)
    s2=hs.signals.Signal2D(im2)
    for i in range(len(s1r.axes_manager.signal_axes)):
        s1.axes_manager.signal_axes[i].scale=s1r.axes_manager.signal_axes[i].scale
        s1.axes_manager.signal_axes[i].units=s1r.axes_manager.signal_axes[i].units
        
        s2.axes_manager.signal_axes[i].scale=s2r.axes_manager.signal_axes[i].scale
        s2.axes_manager.signal_axes[i].units=s2r.axes_manager.signal_axes[i].units
    
    #Get pixel size from dm files
    xaxis,yaxis=s2.axes_manager.signal_axes
    a_spacing=xaxis.scale #used globaly
    
    print("Rebinned image side 1", s1)
    print("Rebinned image side 2", s2)
    print("Rebinned pixel spacing =",a_spacing, xaxis.units)

    #inspect cropped images
    if plot_cropped:
        s1.plot()
        s2.plot()

    return (s1, s2)





def make_square_img(img, is_signal = False):
    """
    crops and image to be square
    """
    if is_signal:
        s=img
        y1,x1=s.data.shape

        if y1==x1:
            pass
        elif y1>x1:
            s=s.isig[:,(y1-x1):]
        else:
            s=s.isig[(x1-y1):,:]
        
        return s
    else:
        y1,x1=img.shape

        if y1==x1:
            pass
        elif y1>x1:
            img=img[(y1-x1):,:]
        else:
            img=img[:,(x1-y1):]
            
        return(img)

    
def crop_to_same_size(im1,im2, is_signal=False):
    """
    crops and image to be the same size
    crop signals to have same size
    """
    if is_signal:
        
        s1=im1
        s2=im2
        
        y1,x1=s1.data.shape
        y2,x2=s2.data.shape

        if x1==x2:
            pass
        elif x1>x2:
            s1=s1.isig[(x1-x2):,:]
        else:
            s2=s2.isig[(x2-x1):,:]

        if y1==y2:
            pass
        elif y1>y2:
            s1=s1.isig[:,(y1-y2):]
        else:
            s2=s2.isig[:,(y2-y1):]
    else:
        s1,s2=im1.copy(),im2.copy()
        
        y1,x1=s1.shape
        y2,x2=s2.shape

        if x1==x2:
            pass
        elif x1>x2:
            s1=s1[:,(x1-x2):]
        else:
            s2=s2[:,(x2-x1):]

        if y1==y2:
            pass
        elif y1>y2:
            s1=s1[(y1-y2):,:]
        else:
            s2=s2[(y2-y1):,:]
            
    return(s1,s2)


def rebin_and_crop_dm (s1_orig, s2_orig, rebin_factor, plot_original = False, plot_cropped = True):
    """
    Takes two hyperspy signals, rebins them, and crops to have equal sizes.
    
    Returns
    (s1, s2) : tuple of hyperspy signals
        The rebinned and cropped images.
    """
    
    
    #inspect dimensions
    print("Original image side 1", s1_orig)
    print("Original image side 2", s2_orig)

    #check scales
    xaxis,yaxis=s1_orig.axes_manager.signal_axes
    a_spacing1=xaxis.scale
    xaxis,yaxis=s2_orig.axes_manager.signal_axes
    a_spacing2=xaxis.scale

    if a_spacing1 == a_spacing2:
        print("Original pixel spacing =",a_spacing2, xaxis.units)
    else:
        raise(Exception("Images have different magnifications"))

    #plot images
    if plot_original:
        s1_orig.plot()
        s2_orig.plot()
        
    print("\n*****\n")
    #rebin
    s1=s1_orig.rebin(scale=(rebin_factor,rebin_factor))/rebin_factor**2
    s2=s2_orig.rebin(scale=(rebin_factor,rebin_factor))/rebin_factor**2

    #crop images to have same size
    s1,s2 = crop_to_same_size(s1,s2, is_signal=True)

    #Get pixel size from dm files
    xaxis,yaxis=s2.axes_manager.signal_axes
    a_spacing=xaxis.scale #used globaly
    
    print("Rebinned image side 1", s1)
    print("Rebinned image side 2", s2)
    print("Rebinned pixel spacing =",a_spacing, xaxis.units)

    #inspect cropped images
    if plot_cropped:
        s1.plot()
        s2.plot()

    return (s1, s2)


def rebin_and_smooth_sig(s, sigma=10, rebin_factor=4):
    """
    smooths and rebins a hyperspy signal
    
    Returns
    signal : hyperspy signal
        
    """
    ss=s.copy()

    #gaussian filter
    ss.data=gfilter(s.data, sigma=sigma)
    
    #rebin images
    ss=s.rebin(scale=(rebin_factor_rec,rebin_factor_rec))

    #correct intensity
    ss=ss/rebin_factor_rec**2
    
    print(ss)
    return(ss)








    
    
def sanity_check_projections (tana, xtilts, xtilt_best = 0, theta_best = 0, bracket=[-np.pi/2, +np.pi/2], plot_results=False):
    i_s = range(len(xtilts))
    num = 2
    combinations = list(itertools.combinations(i_s, num))

    err_fns=[]
    err_fns_th0 = []
    err_fns_xt0 = []

    print("Error for each combination of projections, if we assume theta == 0 and miss-tilt == 0")

    for comb in combinations:
        i,j = comb
        tana1=tana[i]
        tana2=tana[j]
        xtilt1=xtilts[i]
        xtilt2=xtilts[j]
        # err_fns.append(build_error_fn(tana1,tana2,xtilt1,xtilt2)) #unused
        err_fns_th0.append(build_error_fn(tana1,tana2,xtilt1,xtilt2, theta=theta_best))
        err_fns_xt0.append(build_error_fn(tana1,tana2,xtilt1,xtilt2, xtilt=xtilt_best))

        err = ideal_projection_check(tana1, tana2, xtilt1, xtilt2) #does not consider additional parameters
        print(comb, "error: %.4f"%err)

    absolute=False

    print("\n1D root finding solutions")
    #miss-tilt estimates
    xtilt_0s=[]
    for fn in err_fns_th0:
        root = find_root_1var(fn, bracket, args=(absolute,), plot_fn=plot_results)
        xtilt_0s.append(root)
    print("Expect miss-tilt between %.3f and %.3f degrees"%(np.degrees(np.min(xtilt_0s)),np.degrees(np.max(xtilt_0s))))
    
    print(np.degrees(xtilt_0s))
    
    #theta estimates
    thetas=[]
    for fn in err_fns_xt0:
        try:
            root = find_root_1var(fn, bracket, args=(absolute,), plot_fn=plot_results)
        except:
            root=32202
            print("ERROR: no theta solution found. Inserting value 32202 rad")
        thetas.append(root)
    print("Expect theta between %.3f and %.3f degrees"%(np.degrees(np.min(thetas)),np.degrees(np.max(thetas))))
    
    print(np.degrees(thetas))

    


    
    


    

    



def pad_to_same_size(im1,im2, is_phasemaps=False, mode='constant'):
    """
    pads images or phasemaps to have same size. All padding is added at low ends of coordinates.
    """
    if is_phasemaps:
        
        s1=im1
        s2=im2
        
        y1,x1=s1.mask.shape
        y2,x2=s2.mask.shape

        if x1==x2:
            pass
        elif x1>x2:
            padx=x1-x2
            s2=s2.pad(((0,0),(padx,0)), mode=mode)
        else:
            padx=x2-x1
            s1=s1.pad(((0,0),(padx,0)), mode=mode)

        if y1==y2:
            pass
        elif y1>y2:
            pady=y1-y2
            s2=s2.pad(((pady,0),(0,0)), mode=mode)
        else:
            pady=y2-y1
            s1=s1.pad(((pady,0),(0,0)), mode=mode)
            
    else:
        s1,s2=im1.copy(),im2.copy()
        
        y1,x1=s1.shape
        y2,x2=s2.shape

        if x1==x2:
            pass
        elif x1>x2:
            padx=x1-x2
            s2=np.pad(s2,((0,0),(padx,0)), mode=mode)
        else:
            padx=x2-x1
            s1=np.pad(s1,((0,0),(padx,0)), mode=mode)

        if y1==y2:
            pass
        elif y1>y2:
            pady=y1-y2
            s2=np.pad(s2,((pady,0),(0,0)), mode=mode)
        else:
            pady=y2-y1
            s1=np.pad(s1,((pady,0),(0,0)), mode=mode)
            
    return(s1,s2)



    
