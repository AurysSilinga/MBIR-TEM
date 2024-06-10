# -*- coding: utf-8 -*-
# Aurys Silinga, 2024
#

"""
Utility functions for pyramid(by)AS
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyramid as pr

import scipy.optimize as op
from skimage.transform import rescale
import skimage.transform as sktr

def pad_images_to_same_size(im1, im2, mode='constant'):
    """
    Takes two 2-dimensional numpy arrays and pads them to have the same dimensions. Padding attemps to preserve the image centre.
    """
    dy1,dx1=im1.shape
    dy2,dx2=im2.shape
        
    pady=np.abs(dy2-dy1)
    padtop=pady//2
    padbottom=pady//2
    if pady%2 == 1:
        padbottom=padbottom+1
        
    padx=np.abs(dx2-dx1)
    padleft=padx//2
    padright=padx//2
    if padx%2 == 1:
        padright=padright+1
    
    padding1 = [(0,0),(0,0)] 
    padding2 = [(0,0),(0,0)]    
    if dy2 > dy1:
        padding1[0]=(padtop,padbottom)
    else:
        padding2[0]=(padtop,padbottom)
    if dx2 > dx1:
        padding1[1]=(padleft,padright)
    else:
        padding2[1]=(padleft,padright)
        
    im1_padded = np.pad(im1, padding1, mode=mode)
    im2_padded = np.pad(im2, padding2, mode=mode)
    
    return(im1_padded, im2_padded)


def matshow_n(ims, labels=None, save=False, file_type='.tif', origin='upper', plot_cbar=False,**kwargs):
    """
    plot all images in ims one-by-one.
    can save data image as a file.
    """
    if labels is None:
        labels=range(len(ims)+1)
        labels=[str(i) for i in labels]
    
    for i in range(len(ims)):
        plt.matshow(ims[i],origin=origin,**kwargs)
        plt.title(labels[i])
        if plot_cbar:
            plt.colorbar()
        plt.show()
        
        if save:
            img=Image.fromarray(ims[i])
            img.save(labels[i]+file_type)
            


def get_max_ang(field_in, verbose=False):
#return might be wrong for plotting?
    """
    find the angle between nearest neighbours for each pixel
    
    """
    field=np.pad(field_in, ((0,0),(1,1),(1,1),(1,1)), mode='reflect')

    reduced_shape=np.array(field.shape[1:])-2

    def get_ang(z1,y1,x1,z2,y2,x2):
        v1=[z1,y1,x1]
        v2=[z2,y2,x2]
        norm1=np.linalg.norm(v1,None)
        norm2=np.linalg.norm(v2,None)
        if norm1==0 or norm2==0:
            return 0
        else:
            cos=np.dot(v1,v2)/norm1/norm2
            cos=np.where(cos>1, 1.,cos)
            cos=np.where(cos<-1, -1.,cos)
            return(np.degrees(np.arccos(cos)))

    vec_ang=np.vectorize(get_ang, otypes=['float64'])
    
    fieldz,fieldy,fieldx=field
    v_field=(fieldz,fieldy,fieldx)
    
    #6 nearest neighbours in 3D
    #get_ang(v_field[0:1,:,:],v2[1:2,:,:])
    angs=[]
    ang=vec_ang(*field[:,0:-2,1:-1,1:-1],*field[:,1:-1,1:-1,1:-1])
    angs.append(ang)
    ang=vec_ang(*field[:,1:-1,0:-2,1:-1],*field[:,1:-1,1:-1,1:-1])
    angs.append(ang)
    ang=vec_ang(*field[:,1:-1,1:-1,0:-2],*field[:,1:-1,1:-1,1:-1])
    angs.append(ang)
    ang=vec_ang(*field[:,2:,1:-1,1:-1],*field[:,1:-1,1:-1,1:-1])
    angs.append(ang)
    ang=vec_ang(*field[:,1:-1,2:,1:-1],*field[:,1:-1,1:-1,1:-1])
    angs.append(ang)
    ang=vec_ang(*field[:,1:-1,1:-1,2:],*field[:,1:-1,1:-1,1:-1])
    angs.append(ang)

    angs=np.array(angs)
    
    max_ang=np.max(angs, axis=0)
    
    if verbose:
        print('shape to return',reduced_shape)
        print('padded field shape',field.shape)
        print('max ang shape',max_ang.shape)
        print("max ang =", np.max(max_ang))
        
    return(max_ang)
    
    
def slice_on_axis(a, axis, start, end, step=1):
    """
    Perform slice only on one specified axis. Axis can be specified dynamically.
    """
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(start, end, step)
    return a[tuple(sl)]

def get_mask_shifts_in_range(dim, radius=1):
    """
    For an n-dimensional array define the slices that represent all shifts withing distance 'step'.
    
    dim: int
        number of dimensions in a masked image.
    radius: float, default: 1
        maximum shift distance is int(radius)+1.
        
    returns: array of slice objects, shape: [[slice,...,slice],...,[slice,...,slice]]
        When applied to an n-dimensionl image padded by padwidth = step, 
        this generates all possible images reachable by shifting the original image by distance < 'step'.
    """

    if type(radius) is not int:
        step=int(radius)+1
    else:
        step=radius
    axis_ranges = [np.arange(-step, step+1, 1)]*dim
    coords=np.meshgrid(*axis_ranges)

    
    #find all valid nth-neighbours and calculate appropriate shifts.
    slices=[]
    coords = np.array([np.ravel(coord) for coord in coords]) #all possible coordinate combinations
    for i in range(coords.shape[1]):
        if np.sqrt(np.dot(coords[:,i], coords[:,i])) <= radius: # if within range
            slice_indexes=[(step,-step)]*dim
            shifts=coords[:,i]
            for j,shift in enumerate(shifts):
                low_i, high_i = slice_indexes[j]
                low_i+=shift
                high_i+=shift
                if low_i == 0:
                    low_i = None
                if high_i == 0:
                    high_i = None
                slice_indexes[j] = (low_i,high_i)
            multi_slice = tuple([slice(*start_stop) for start_stop in slice_indexes])
            slices.append(multi_slice)
            
    return (slices)
    
def find_outer_shell_pixels (wrongs, radius=1, internal=False, outside_image_is_wrong = True):
    """
    Find all pixels that are 'radius' pixels away from the surface. 
    
    Attributes:
    -----------
    wrongs: numpy array, bool.
        Mask that is True where pixels are wrong.
    radius: float, default: 1
        pixels within 'radius' of the surface are selected.
    wrong_pixel_value: float, default: 0
    outside_image_is_wrong: bool, default: True
        if image edges are considered as surface edges.
    """
    
    if type(radius) is not int:
        step=int(radius)+1
    else:
        step=radius
    dim = wrongs.ndim
    filled = np.bitwise_not(wrongs)
    if outside_image_is_wrong:
        padding_value = False
    else:
        padding_value = True
    padded_filled = np.pad(filled, step, constant_values=padding_value)
    axis_ranges = [np.arange(-step, step+1, 1)]*dim
    coords=np.meshgrid(*axis_ranges)
    
    #find all valid nth-neighbours
    neighbours = np.full(filled.shape, False)
    slices=get_mask_shifts_in_range(wrongs.ndim, radius=radius)
    for multi_slice in slices:
        shifted_filled = padded_filled[multi_slice]

        #identify pixels that are within distance 'step'
        if internal:
            to_fill = np.bitwise_and(filled, np.bitwise_not(shifted_filled)) # previously filled and on the outer edge.
        else:
            to_fill = np.bitwise_and(shifted_filled, np.bitwise_not(filled)) # shifted and not previously filled
        neighbours[to_fill]=True
    return (neighbours)

def fill_masked_image(img, wrongs, radius=2, shell_width = 1, 
                               iterations=-1, current_iteration=0, wrong_pixel_value=None):
    """
    Every wrong pixel is replaced by the average of correct pixels within 'radius'.
    Algorithm extends correct parts of the image one shell at a time, recursively.
    Should work in any numer of dimensions but is only tested for 2 and 3.
    Execution for image with N pixels execution scales as O(N * radius^2 / shell_width)
    
    Attributes
    ----------
    image : numpy array, float
    wrongs: numpy array, bool.
        Mask that is True where pixels are wrong.
    radius: float, default: 2
        distance in pixels from the centre of a wrong pixel. 
        Valid pixels within radius are used to calculate the replacement value.
    shell_width: float, default: 1
        how many layers from the surface to fill in one iteration. Need 'radius' >> 'shell_width' to avoid streaking.
    iterations: int, default: -1
        how many layers to process before terminating. iterations = -1 terminates once all wrong pixels have been fixed
    wrong_pixel_value: float, default: None
        what value to give wrong pixels that were not filled. wrong_pixel_value = None assigns max(img)+1 by default.
    """
    img=img.copy()
    if wrong_pixel_value is None:
        wrong_pixel_value = np.nanmax(img)+1
    img[wrongs] = wrong_pixel_value
    if type(radius) is not int:
        step=int(radius)+1
    else:
        step=radius
    padded_img = np.pad(img, step)
    filled = np.bitwise_not(wrongs)
    padded_filled = np.pad(filled, step)
    hit_counter = np.zeros(filled.shape)
    extension = np.zeros(filled.shape)
    
    #shift image one space in each direction and fill wrong pixels with average of their correct neighbours
    slice_list = get_mask_shifts_in_range (img.ndim, radius=radius)
    for slices in slice_list:
        shifted_filled = padded_filled[slices]
        shifted_img = padded_img[slices]

        #identify pixels that need to be filled and fill then
        to_fill = np.bitwise_and(shifted_filled, np.bitwise_not(filled)) # shifted and not previously filled
        hit_counter[to_fill]+=1
        extension[to_fill]+=shifted_img[to_fill]
    
    #average pixels with multiple contributions
    extension[hit_counter>0]=extension[hit_counter>0]/hit_counter[hit_counter>0]
    nearest_neighbours = find_outer_shell_pixels (wrongs, radius=shell_width)
    img[nearest_neighbours]=extension[nearest_neighbours]
    wrongs = (img == wrong_pixel_value)
    
    if (True not in wrongs) or ((current_iteration+1 >= iterations) and iterations!=-1):
        return (img, wrongs)
    else:
        return fill_masked_image(img, wrongs, step, shell_width, iterations=iterations, 
                                          current_iteration=current_iteration+1, wrong_pixel_value=wrong_pixel_value)
    
def erode_masked_image(img, wrongs, radius=1, wrong_pixel_value=0, erode_edges=True):
    """
    Remove the surface layer of a masked image and replace with 'wrong_pixel_value'.
    'img' can have any number of dimensions.
    
    Attributes:
    -----------
    image : numpy array, float
    wrongs: numpy array, bool.
        Mask that is True where pixels are wrong.
    radius: float, default: 1
        pixels within 'radius' of the surface are removed.
    wrong_pixel_value: float, default: 0
        replace wrong pixels with this value in the image.
    erode_edges: bool, default: True
        if pixels outside the image are considered wrong.
    """
    surface_shell = find_outer_shell_pixels(wrongs, radius=radius, internal=True, outside_image_is_wrong=erode_edges)
    img=img.copy()
    img[surface_shell] = wrong_pixel_value
    wrongs = np.bitwise_or(wrongs, surface_shell)
    return (img, wrongs)
    
    
def rebin_img(img, rebin_factor=2, crop=False):
    """
    Rebins the image by averaging 4 neighbouring pixels.
    factor=2 rebins once.
    crop=True to mimic hyperspy crop behaviour
    """
    img_rb=img.copy()
    for i in range(int(np.log2(rebin_factor))):
        
        #if image size not divisible by 2, extend or trim the edge
        dy,dx=img_rb.shape
        if dy%2 ==1:
            if crop:
                img_rb=img_rb[:-1,:]
            else:
                img_rb=np.pad(img_rb,((0,1),(0,0)), mode='edge')
        if dx%2 ==1:
            if crop:
                img_rb=img_rb[:,:-1]
            else:
                img_rb=np.pad(img_rb,((0,0),(0,1)), mode='edge')
        
        tl=img_rb[0::2,0::2]
        tr=img_rb[0::2,1::2]
        bl=img_rb[1::2,0::2]
        br=img_rb[0::2,1::2]
        img_rb=np.mean([tl,tr,bl,br], axis=0)

    return (img_rb)
    

def flood_fill(point, results, allowed_space, ylimits, xlimits):
    """
    Space filling arlgorithm. Finds all squares connected to a starting point.
    
    Atributes:
    ----------
    point: tuple of int
        starting point.
    results: numpy array of False
        pixels are marked as True if they connect.
    allowed_space: numpy arrray of bool
        False means pixel is not accessible, True means it is fillable. 
        Must have same size as 'results' array
    xlimits and ylimits: tuple of int:
        minimum and maximum allowed coordinate values, respectively.
    """
    py,px=point
    if py > ylimits[1] or py < ylimits[0] or px > xlimits[1] or px < xlimits[0]: #if outside valid range
        return (0)
    if not allowed_space[point] or results[point]: #if previously visited or invalid, do nothing
        return (0)
    else:
        results[point] = True
        flood_fill((py+1, px), results, allowed_space, ylimits, xlimits)
        flood_fill((py-1, px), results, allowed_space, ylimits, xlimits)
        flood_fill((py, px+1), results, allowed_space, ylimits, xlimits)
        flood_fill((py, px-1), results, allowed_space, ylimits, xlimits)
        return (1)
        
def span_fill(point, results, allowed_space, ylimits, xlimits):
    """
    Space filling arlgorithm. Finds all squares connected to a starting point. 
    Optimised to scan horizontal lines without recursive calls.
    
    Atributes:
    ----------
    point: tuple of int
        starting point.
    results: numpy array of False
        pixels are marked as True if they connect.
    allowed_space: numpy arrray of bool
        False means pixel is not accessible, True means it is fillable. 
        Must have same size as 'results' array
    xlimits and ylimits: tuple of int:
        minimum and maximum allowed coordinate values, respectively.
    """
    def not_valid(point):
        py,px = point
        if py > ylimits[1] or py < ylimits[0] or px > xlimits[1] or px < xlimits[0]: #if outside valid range
            return (True)
        elif not allowed_space[point] or results[point]: #if previously visited or invalid, do nothing
            return (True)
        else:
            return (False)
    
    if not_valid(point):
        return 0
    else:
        results[point] = True
        py,px=point

        #scan left
        for sx in range(xlimits[0],px)[::-1]:
            scan_point = (py,sx)
            if not_valid(scan_point):
                break
            else:
                results[scan_point]=True
                span_fill((py+1, sx), results, allowed_space, ylimits, xlimits)
                span_fill((py-1, sx), results, allowed_space, ylimits, xlimits)

        #scan right
        for sx in range(px+1,xlimits[1]+1):
            scan_point = (py,sx)
            if not_valid(scan_point):
                break
            else:
                results[scan_point]=True
                span_fill((py+1, sx), results, allowed_space, ylimits, xlimits)
                span_fill((py-1, sx), results, allowed_space, ylimits, xlimits)

        span_fill((py+1, px), results, allowed_space, ylimits, xlimits)
        span_fill((py-1, px), results, allowed_space, ylimits, xlimits)
        return (1)
        
        
def sktr_translate(img, dx=0, dy=0, **kwargs):
    """
    intuitive translation. dx=5, moves the image in +x direction by 5 pixels.
    """
    dtype=img.dtype
    trans=sktr.EuclideanTransform(translation=(-dx, -dy),**kwargs)
    return (sktr.warp(img,trans).astype(dtype))
    
    
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    values for a 2D gaussian rotated by 'theta' (in radians).
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g

def gaussian_moments(xy, data_2d, theta=0, is_error_function=False, plot_results=False):
    """
    Returns (amplitude, x0, y0, width_x, width_y, theta, offset)
    the gaussian parameters of a 2D distribution by calculating its moments.
    theta: float, default: 0
        angle in radians of counter-clockwise rotation of the gaussian. used to improve paratmeter estimation
    is_error_function: bool, default: True
        if True, assumes that the gaussian peak is at z=0 and the edges are at some value > 0. 
        otherwise assumes that the gaussian has a peak at z>0 and amplitude>0
    """
    x, y = xy
        
    if is_error_function: #peak is at 0
        amp = -np.max(data_2d)
        offset = -amp
        data = np.max(data_2d) - data_2d
    else:
        offset=np.min(data_2d)
        amp=np.max(data_2d) - offset
        data = data_2d - offset
            
    if theta != 0: #rotate axes of the gaussian
        data=sktr.rotate(data, np.degrees(theta))
        
    #centre of mass
    total = np.sum(data)
    x0 = np.sum(x*data)/total
    y0 = np.sum(y*data)/total
        
    #variance of input PDF
    col = data[:, int(x0)] #central collumn
    col_indexes =y[:, int(x0)]
    width_y = np.sqrt(np.sum((col_indexes-y0)**2*col)/np.sum(col))
    y_range= np.max(y) - np.min(y)
    if width_y < 0.1*y_range/2 or width_y > 10*y_range/2:
        width_y = y_range/2
    if plot_results:
        plt.figure()
        plt.plot(col_indexes, col, label=f"central collumn, x0={x0:.2f}, sigma={width_y:.2f}")

    row = data[int(y0), :]
    row_indexes =x[int(y0), :]
    width_x = np.sqrt(np.sum((row_indexes-x0)**2*row)/np.sum(row))
    x_range= np.max(x) - np.min(x)
    if width_x < 0.1*x_range/2 or width_x > 10*x_range/2:
        width_x = x_range/2
    if plot_results:
        plt.plot(row_indexes, row, label=f"central row, y0={y0:.2f}, sigma={width_x:.2f} ")
        plt.title("CoM and variance estimates before fitting")
        plt.legend()

    
    return (amp, x0, y0, width_x, width_y, theta, offset)

def fit_2D_gaussian_blob(xy, data_2d, initial_guess=None, bounds=None, gauss_fn=gaussian_2d,
                        theta=0, plot_results=True, is_error_function=False, verbose=False, labels=["",""]):
    """
    Fits a 2D gaussian function to a dataset. 
    if 'initial_guess is None' determines first two moments to estimate starting conditions.
    """
    x, y = xy
    def gaussian_flattened(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        return gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset).ravel()
    
    if initial_guess is None:
        initial_guess=gaussian_moments(xy, data_2d, theta=theta, is_error_function=is_error_function, plot_results=verbose)
        if verbose:
            print("Estimated starting parameters: amplitude, x0, y0, sigma_x, sigma_y, theta, offset")
            print(initial_guess)
            
    if bounds is None:
        #apply loosest possible bounds with physical meaning
        xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
        if is_error_function:
            bounds=[(-np.inf, xmin, ymin, 0     , 0     , np.radians(-90), 0      ), 
                    (0      , xmax, ymax, np.inf, np.inf, np.radians(90) , np.inf)]
        else:
            bounds=[(-np.inf, xmin, ymin, 0     , 0     , np.radians(-90), -np.inf), 
                    ( np.inf, xmax, ymax, np.inf, np.inf, np.radians(90) ,  np.inf)]
    
    try:
        popt, pcov = op.curve_fit(gaussian_flattened, (x, y), data_2d.ravel(), p0=initial_guess, bounds=bounds)
    except Exception as EXC:
        plt.matshow(data_2d)
        plt.title("Could not fit a Gaussian distribution to this data")
        raise(EXC)
        
    if plot_results:
        data_fitted = gaussian_2d((x, y), *popt)
        extent=np.min(x), np.max(x), np.min(y), np.max(y)
        plt.figure()
        plt.imshow(data_2d, extent = extent, cmap='nipy_spectral', origin='upper')
        plt.colorbar()
        plt.contour(x, y[::-1], data_fitted, 4, colors='w') #origin is selected by invering y axis
        plt.title("Error function (rainbow) with Gaussian fit (white)")
        plt.xlabel(labels[1])
        plt.ylabel(labels[0])
    
    return(popt,pcov)
    
def get_distance_grid(shape, centre=None):
    """
    Takes an array shape, and calculates the geometric distance of each pixel from the centre
    """
    axis_ranges=[]
    for i,axis_range in enumerate(shape):
        if centre is None:
            half_length = axis_range/2-0.5
            axis=np.arange(-half_length, half_length+1, step=1) #centre is 0
        else:
            axis=np.arange(axis_range, step=1)-centre[i]
        axis_ranges.append(axis)
    index_grids = np.meshgrid(*axis_ranges, indexing='ij') #this is [gz, gy, gx]

    distance=np.zeros(shape)
    for coord_grid in index_grids:
        distance += coord_grid**2
    distance = np.sqrt(distance)
    return(distance)
    
    
def pad_to_even(array, mode='constant', invert=False, square=False): #'edge'
    #TODO move to the alignment tab where this niche thing is needed
    """
    pads an array to have even dimensions along every axis. 
    if square=True, makes the array square
    
    if invert = True:
        returns (padded_array, unpadding_slice) 
    else:
        returns (padded_array)
    """
    pad_val = []
    for dim in array.shape:
        if square:
            max_dim=np.max(array.shape)
            max_dim+= max_dim%2 #make divisible by 2
            pad_val.append(max_dim-dim)
        else:    
            pad_val.append(dim%2)
    padding=tuple([(val,0) for val in pad_val])
    padded_array=np.pad(array, padding, mode=mode)
    inversion_multislice=tuple([slice(val,None) for val in pad_val])
    
    if invert:
        return (padded_array, inversion_multislice)

    else:
        return(padded_array)
        
def rescale_complex(img, scale):
    """
    Skimage rescale function, but works on complex numbers
    returns: rescale(image, scale)
    """
    r= np.real(img)
    c=np.imag(img)
    return(rescale(r,scale)+1j*rescale(c,scale))