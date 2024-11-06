# -*- coding: utf-8 -*-
# Aurys Silinga, 2024
#

"""
Simulation functions for pyramid(by)AS
"""

import pyramid as pr
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack as fft #old scipy version
import scipy.optimize as op
import skimage.filters as skfl
import skimage.transform as sktr
import copy

from .util import pad_to_even, rescale_complex, fill_masked_image
from .alignment import find_edges


def find_voxel_coords(mask_3d, position_zyx=None, mask_edge=0, plot_results=False):
    """
    outputs the coordinates (z,y,x) of a voxel in the 3D mask.
    if coords_zyx=None, then returns the center of mass
    else if coords = [None,None,10] then returns (centre, centre, 10)
    """
    coords=position_zyx
    dim = mask_3d.shape
    pixel_pos=[]
    if coords is None:
        # do Centre of Mass
        coord_ranges=[]
        for i in range(len(dim)):
            coord_ranges.append(np.arange(dim[i]))
        index_grids = np.meshgrid(*coord_ranges, indexing='ij')
        
        #calculate CoM
        mask = np.where(mask_3d, 1, 0) #make numerical
        mask[...,:mask_edge]=0
        for grid in index_grids:
            com = np.sum(grid*mask)/np.sum(mask)
            pixel_pos.append(round(com))
    else:
        for i,axis in enumerate(coords):
            if axis is None:
                axis=dim[i]//2
            pixel_pos.append(int(axis))
    if mask_3d.ndim != len(pixel_pos):
        raise TypeError("Mask and pixel position need to have the same number of dimensions")
    
    if plot_results:
        plt.figure()
        ax1 = plt.subplot(211)
        plt.imshow(np.sum(mask_3d, axis=0), origin='lower')
        #plt.title("mask z-axis sum with selected point showed")
        plt.plot([pixel_pos[2]],[pixel_pos[1]], 'ro')
        plt.ylabel("y")
        plt.xlabel("x")

        ax2 = plt.subplot(212, sharex=ax1)
        plt.imshow(np.sum(mask_3d, axis=1), origin='lower')
        #plt.title("mask y-axis sum with selected point showed")
        plt.plot([pixel_pos[2]],[pixel_pos[0]], 'ro')
        plt.ylabel("z")
        plt.xlabel("x")
        
        plt.suptitle("z- and y-axis projections showing selected point")
        plt.show()
        
    return(tuple(pixel_pos))

def bayesian_diagnostics(data, mag_rec, cost_f, voxel_position_zyx=None, 
                       verbose=False, max_iter=1000, plot_results=True):
    """
    Performs the inverse minimisation problem and solves for 
    probability density function (PDF) of magnetisation vector at a singe pixel.
    See Jan Caron thesis section 4.4 for mathematical definitions
    
    Calculates the spatial resolution based on PDF full-width-half-maximum
    Also calculates the error on the magnitude caused by residuals between experimental and calculated phasemaps.
    """
    
    if voxel_position_zyx is None:
        print("Determining test voxel position automatically.")
        mask=data.mask
        voxel_position_zyx = find_voxel_coords(mask, position_zyx=None, plot_results=True)
        
    dimz,dimy,dimx=mag_rec.dim
    position=voxel_position_zyx
    
    fwhm_vec = []
    error_vec = []
    diagnostic_vec = []
    
    #calculate gain per pixel
    backup_cache=copy.deepcopy(cost_f.fwd_model.ramp.param_cache)
    for vec_component in range(3): #(x,y,z)
        cost_f.fwd_model.ramp.param_cache=copy.deepcopy(backup_cache)
        diagnostic = pr.Diagnostics(mag_rec, cost_f, verbose=False, max_iter=max_iter)
        diagnostic.pos = (vec_component, *position)
        fwhm_component_xyz, fit_param1, fit_param2 = diagnostic.calculate_fwhm(plot=plot_results)
        fwhm_pix = np.array(fwhm_component_xyz)/mag_rec.a
        G_row=diagnostic.gain_row #performs minimisation using original forward model and changes the ramp cache
        cost_f.fwd_model.ramp.param_cache=copy.deepcopy(backup_cache) #reset cache before simulating phasemaps
    
        #assume that error for each pixel is experimental phase - simulated phase, and use that to find error on amplitude
        phasemaps_diff = data.create_phasemaps(mag_rec, difference=True, ramp=cost_f.fwd_model.ramp)
        #calculate rms difference only using valid pixels
        phase_diff_std=np.std(np.concatenate([p.phase[p.confidence>0] for p in phasemaps_diff])) 
        # err_vec = G_row * phase_vec_calc #how much error each individual pixel creates on selected voxel
        err = np.sqrt(np.dot(G_row,G_row)*phase_diff_std*phase_diff_std) #total error by assuming pixel rms error is standard deviation.
        
        fwhm_vec.append(fwhm_pix)
        error_vec.append(err)
        diagnostic_vec.append(diagnostic)
        if verbose:
            s="xyz"
            print("finished calculating for", s[vec_component],"component")
    
    #perform error propagation
    #fwhm_vec is [(dx_x, dy_x, dz_x), (dx_y, dy_y, dz_y), (dx_z, dy_z, dz_z)]
    mag_xyz = mag_rec.field[:,position[0],position[1],position[2]]
    amplitude = np.sqrt(np.dot(mag_xyz, mag_xyz))
    fwhm_arr = np.array(fwhm_vec) #reverse order from usual numpy
    dx_vec2=np.sum((fwhm_arr[:,0]*mag_xyz/amplitude)**2)
    dy_vec2=np.sum((fwhm_arr[:,1]*mag_xyz/amplitude)**2)
    dz_vec2=np.sum((fwhm_arr[:,2]*mag_xyz/amplitude)**2)
    fwhm_xyz = np.sqrt((dx_vec2, dy_vec2, dz_vec2)) #propagated spatial error

    error_vec = np.array(error_vec)
    error_vec = error_vec*mag_xyz/amplitude
    error_total = np.sqrt(np.dot(error_vec, error_vec)) #propagate errors for amplitude
    
    if verbose:
        print("Voxel position:", position)
        print(f"""Magnetisation vector is:
        (M_x = {mag_xyz[0]:.2e} +/- {error_vec[0]:.2e} T,
         M_y = {mag_xyz[1]:.2e} +/- {error_vec[1]:.2e} T,
         M_z = {mag_xyz[2]:.2e} +/- {error_vec[2]:.2e} T)""")
        print(f"Amplitude: {amplitude:.3e} +/- {error_total:.3e} T")
        print(f"Spatial resolution (dx, dy, dz): {fwhm_xyz[0]:.1f}, {fwhm_xyz[1]:.1f}, {fwhm_xyz[2]:.1f} pixels")
        print(f"Pixel spacing: {data.a:.2f} nm")
        
    return(error_total, fwhm_xyz, tuple(diagnostic_vec))

def plot_error_gain_maps(data, diagnostic_results, component='x'):
    """
    display gain maps after calculating the Bayesian error estimates
    
    Returns: None
    """

    s='xyz'
    component_index = s.find(component)
    diagnostic_results
    amp_resolution, spatial_resolution, diagnostic_datasets  =  diagnostic_results
    diagnostic = diagnostic_datasets[component_index]
    gain_maps=diagnostic.get_gain_maps()
    for i in range(len(gain_maps)):
        gm=gain_maps[i]
        tilt=np.degrees(data.projectors[i].tilt)
        gm.plot_phase(title=f"Tilt = {tilt:.2f} deg, voxel {diagnostic.pos[1:]}",unit="T/rad")
        
        
def get_spherical_shells(dim, shell_width = 1, min_radius=0, max_radius=None):
    """
    Given a n-dimensiona array of (e.g. dim = (10,20,20)) finds all the spherical shells centred in the middle of the volume
    
    Returns: list of ndarray(type==bool)
    """
    #create coordinate grids with 0 at the centre  
    axis_ranges = []
    for axis_range in dim:
        axis_ranges.append(fft.fftshift(fft.fftfreq(axis_range, d=1/axis_range))) #centre is whatever scipy needs. usually off-centre by 1

    index_grids = np.meshgrid(*axis_ranges, indexing='ij') #this is [gz, gy, gx]
    
    #calculates distances from centre of FFT, which are used to define shells in 3d space
    distance=np.zeros(dim)
    for coord_grid in index_grids:
        distance += coord_grid**2
    distance = np.sqrt(distance)
    
    #find all spherical shells
    shells=[]
    if max_radius is None:
        max_radius=np.max(dim)/2
    
    radii = np.arange(min_radius, max_radius, shell_width)
    #define mask selecting a shell of pixels centered around the centre
    for radius in radii:
        shell_mask= np.bitwise_and((distance >= radius), (distance < radius+shell_width))
        shells.append(shell_mask)
    return(shells)
    
    
def fsc_split_array(field, interpolation_radius=1.5):
    """
    Takes a 2d or 3d array and splits it into two random sub-arrays of the same dimensions. 
    Fills empty pixels after splitting by nearest neighbour interpolation.
    Returns fourier transforms of the sub-arrays
    """
    
    dim = field.shape
    max_dim = np.max(dim)
    rng = np.random.default_rng()
    
    random_selection_3d=rng.integers(2,size=dim) # 1 or 0
    
    wrongs1=random_selection_3d==1
    wrongs2=random_selection_3d==0
    
    field1=field.copy()
    field2=field.copy()
    
    field1[wrongs1]=np.nan
    field2[wrongs2]=np.nan
    
    #fill empty pixels
    field1, temp_wrongs = fill_masked_image(field1, wrongs1, radius=interpolation_radius) 
    field2, temp_wrongs = fill_masked_image(field2, wrongs2, radius=interpolation_radius)
    
    #correct dimensions to have even spatial frequency spacing
    field1, reverse_slice=pad_to_even(field1, invert=True, square=True)
    field2, reverse_slice=pad_to_even(field2, invert=True, square=True)
    
    fft1=fft.fftshift(fft.fftn(field1))
    fft2=fft.fftshift(fft.fftn(field2))
    
    # note, inversion is done like: field1_rec=fft.ifftn(fft.fftshift(fft1))[reverse_slice]
    
    return(fft1, fft2)


def fsc_calculate_correlation(field, fftvol1, fftvol2, scale=2, max_radius=None, min_radius=0, shell_width=1, n_assym=1,
                              pad_values=0, plot_results=False):
    """
    Given two fourier space spheres, calculates the correlation coefficients for spatial frequency shells.
    Uses spatial, and symmetry corrections to calculate effective number of points in a shell.
    scale>1 is used to interpolate the arrays such that pixels are not double-counted and the fsc curve is smoother.
    returns (frequency, correlation_coeficients, n_effective)
    """
    
    #interpolate in-between pixels
    fftvol1=rescale_complex(fftvol1, scale)
    fftvol2=rescale_complex(fftvol2, scale)
    
    if max_radius is None:
        max_radius=np.max(fftvol1.shape)/2
        
    #calculate shells, accounting for dimension change due to interpolation
    shells=get_spherical_shells(fftvol1.shape, min_radius=min_radius, max_radius=max_radius, shell_width=scale*shell_width)
    
    FSC=[]
    ns_effective=[]
    for shell in shells:
        vals1=fftvol1[shell]
        vals2=np.conjugate(fftvol2[shell])
        num=np.sum(vals1*vals2)
        denom=np.sqrt(np.sum(np.abs(vals1)**2) * np.sum(np.abs(vals2)**2))
        FSC.append(np.abs(num/denom))
        
        #size correction
        sizes=[]
        for axis in range(field.ndim):
            left, right = find_edges(field==pad_values, axis=axis)
            sizes.append(right-left)
        D = np.min(sizes) #average object length in one dimension. minimum extent is most representative when tested
        L=np.max(field.shape) #length of space where object is located. all dimensions should be equal
        n=len(vals1)
        ne= n * (3/2*D/L)**2 * 1/2 * 1/n_assym * 1/scale**2
        if ne < 1:
            ne=1
        ns_effective.append(ne)
    
    #calculate the spatial frequency axis
    freq=[]
    freq_full = fft.fftfreq(np.max(fftvol1.shape))
    band_centre = freq_full[shell_width*scale]/2
    radii = np.arange(min_radius, max_radius, shell_width*scale).astype(int) #pixels from centre
    for radius in radii:
        freq.append(freq_full[radius]+band_centre)
    
    if plot_results:
        plt.figure()
        plt.plot(freq, FSC, 'r.-', label="FSC")
        plt.plot(freq, 3/(np.sqrt(ns_effective)+2), 'k-.', label="3 sigma")
        plt.legend()
        plt.xlabel("cycles/pix")
    
    return(freq, FSC, ns_effective)
    
    
def histogram_magnetisation (magdata_rec, range_hist = (1,2), n_bins=100, save_img=False, verbose=True, fit_gauss=True):
    """
    Plots a histogram of magnetisation amplitude and fits a gaussian.
    
    returns: (bin_counts, bin_edges)
    """

    amp = magdata_rec.field_amp.copy()
    amp_distribution = amp[amp>0]

    bin_size = (range_hist[1]-range_hist[0])/n_bins
    

    plt.figure()
    bins, bin_edges, temp_patches = plt.hist(amp_distribution, bins = n_bins, range=range_hist, label=f"Bin size is {bin_size} T")
    plt.xlabel("Magnetisation * $\mu_0$, T")
    plt.ylabel("Number of voxels")
    
    #fit gaussian
    if fit_gauss:
        def gaussian_1D(x, a, b, c):
            return a* np.e**((-1/2)*((x-b)/c)**2)
        fun=gaussian_1D
        x=np.array(bin_edges[:-1])+bin_size/2 #bin centres
        y=bins
        starting_pos = (np.max(bins), np.median(x), 0.1)
        pop, pcov = op.curve_fit(fun, x, y, p0=starting_pos)
        fit_err = np.sqrt(np.diag(pcov))
        plt.plot(x, fun(x,*pop), 'r-', label="Gaussian fit")

    plt.legend()
    plt.tight_layout()
    if save_img:
        plt.savefig("amp bin.png",dpi=200)
    plt.show()
    
    if verbose:
        print("bin_size:", bin_size, "T")
        if fit_gauss:
            print("fitting params (a,b,c) with error:", pop, fit_err)
        print("mean and std:", np.mean(amp_distribution), np.std(amp_distribution))
    
    return (bins, bin_size)
