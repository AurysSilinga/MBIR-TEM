# -*- coding: utf-8 -*-
# Aurys Silinga, 2024
#

"""
Convenient functions for doing 2D and 2D to 3D M reconstructions
"""

import pyramid as pr
from . import alignment as pa
from . import util as pu
from . import reconstruction as pre
from . import tomography as prt

import fpd
import os
import hyperspy.api as hs
import skimage.filters as skfl
import skimage.transform as sktr
import scipy.optimize as op

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle

import astra

from .reconstructionCUDA import *


def load_dm(f_path1, f_path2, fringe_spacing=0, roi_indices=None, plot_input=False, plot_output=False, verbose=False):
    """
    Opens two phase images that are .dm3 files "file1" and "file2".
    Pads them so they are the same size. 
    Rebins to reduce the number of pixels depending on the hologram "fringe_spacing" without loss of spacial resolution.
    Shows interactive region-of-interest used for image alignment.

    fringe_spacing: float (default: 0)
        hologram fringe spacing in nm. images are rebbined such that pixel size is slightly less than fringe_spacing.
    cropping indces: tuple of int (defautl: None):
        (left, right, top, bottom) indices in image1 that define the starting positions of the roi rectangle.
        if None, indices are 10% of image size from the edge.

    Returns: (image1, image2, pix_spacing, roi2D) 
        image1 and image2: (Signal2D) hyperspy signals of file1 and file2 respectively
        pix_spacing: (float) pixels size in nm
        roi2D: (RectangularROI) hyperspy interactive object
    """

    #load dm files
    s1_orig=hs.load(f_path1)
    s2_orig=hs.load(f_path2)

    if verbose:
        print("Loaded",f_path1,"and",f_path2,"\n")
    
    s1, s2, a_spacing = pa.equalise_hspy_signals(s1_orig, s2_orig, fringe_spacing, plot_original = plot_input, plot_cropped = plot_output, verbose=verbose)

    #start the UI
    # make interactive RoI selector
    img=s1.data+s2.data
    s=hs.signals.Signal2D(img)
    s.metadata.General.title = "Select edge detection area"
    s.plot(title="Edge detection area selector")
    if roi_indices is None:
        left,right,top,bottom=s.axes_manager.signal_extent
        left,right,top,bottom=right*0.1,bottom*0.1,right*0.9,bottom*0.9
        print("Adjust the interactive ROI now!")
    else:
        left,right,top,bottom=[int(x) for x in roi_indices]
    rect=hs.roi.RectangularROI(left,right,top,bottom)
    roi2D = rect.interactive(s) 

    return(s1,s2,a_spacing,roi2D)


def identify_sample_edges(s1, s2, roi2D=None, sigma=2, high_threshold=5, hole_filling_radius = 1, erode_width=3, plot_output=True, verbose=False):
    """
    Identifies and plots sample edges. 
    One should trial and error the controll parameters until the correct edge is found.

    roi2D: (RectangularROI)
        interactive ROI object
    erode_width: (int, default: 3)
        masks blurry image edges that may appear is one of the images was rotated
    
    sigma=2 #increase to remove noise, but keep low so edges are preserved
    high_threshold=5 #increase until only one edge remains
    hole_filling_radius=1 #improves smoothness of masked holes but slows down execution

    Returns: (edge1, edge2)
        edge1 and edge2: 2D np.array
    """    
    if roi2D is None:
        roi_joint = np.full(s1.data.shape, True)
    else:
        #define the RoI based on interactive image
        indexes = roi2D.axes_manager.signal_extent
        left,right,top,bottom=[int(x) for x in indexes]
        roi_joint = np.full(s1.data.shape, False)
        roi_joint[top:bottom, left:right] = True
        if verbose:
            print("(left, right, top, bottom):", roi2D.axes_manager.signal_extent)
    
    #fill masked holes to avoid discontinuities
    img1 = pa.fill_raw_dm_image(s1.data, erode_width=erode_width, radius=hole_filling_radius)
    img2 = pa.fill_raw_dm_image(s2.data, erode_width=erode_width, radius=hole_filling_radius)
    
    edges1,edges2 = pa.identify_edges(img1, img2, sigma=sigma, high_threshold=high_threshold, roi=roi_joint, plot_result=False)

    #remove unneeded area
    wrongs=np.logical_not(roi_joint) 
    edges1[wrongs] = 0
    edges2[wrongs] = 0
    
    if plot_output:
        pa.subplots_n((edges1,edges2), labels=("Image 1 edges","Image 2 edges"))
    
    return(edges1, edges2)

def rotate_crop_signal(s1, s2, pre_rotation=0, pre_crop=(slice(None),slice(None)), post_rotation=0, post_crop=(slice(None),slice(None)), 
                       plot_input=False, plot_output=True):
    """
    Take two hyperspy signals. 
    Rotate, crop, rotate again, and crop again.
    Used to crop out the substrate and rotate the image close to the tilt axis.
    
    pre_rotation = 90 #deg, y-to-x direction
    pre_crop = (slice(0,-1),slice(0,-1))
    post_rotation = 0 #deg, y-to-x direction
    post_crop = (slice(0,-1),slice(0,-1)) #(y,x)
    
    Returns: (s1, s2)
        the cropped and rotated signals
    """
    cropped_signals=[]
    for s in (s1,s2):
        s=s.copy()
        im0=s.data.copy()
        s.data=sktr.rotate(s.data, pre_rotation, resize=True)
        im1=s.data.copy()
        s.data=s.data[pre_crop]
        im2=s.data.copy()
        s.data=sktr.rotate(s.data, post_rotation, resize=True)
        im3=s.data.copy()
        s.data=s.data[post_crop]
        im4=s.data.copy()
        cropped_signals.append(s)
    if plot_input:
        pu.subplots_n([im0,im1,im2,im3,im4], labels=["0. Original", "1. R1", "2. R1+C1", "3. R1+C1+R2", "4. R1+C1+R2+C2"],
                      shape=(2,3), title='Steps for image 2')
        plt.axis("off")
        plt.tight_layout()
    if plot_output:
        pu.subplots_n([s.data for s in cropped_signals], labels=["Image1", "Image2"])
        
    return(cropped_signals)

def phase_to_confidence(s1, s2_tr, sigma_mag_rec = 1, median_footprint=np.ones((5,5)), rebin_factor = 2, 
                      phase_img_dtype='f4', erode_depth=3, fill_convolution_radius=2, plot_output=False, verbose=False, save_path=None):
    """
    Make rebinned, smooth phasemaps
    sigma_mag_rec = 1 #sigma of gaussian filter before magnetisation reconstruction 
    median_footprint=np.ones((5,5)) #removes single pixel artefacts
    rebin_factor = 2 # to make total number of voxels less than 10^6
    phase_img_type='f4'
    erode_depth=3 #number of pixels to remove from edge
    fill_convolution_radius=2 #smoothness of filled holes
    # hole indentification was removed
    
    Returns: (mag_phase, el_phase, confidence) (phasemaps)
    """
    pm_mag=(s1-s2_tr)/2
    pm_el=(s1+s2_tr)/2
    
    #Define mask showing which pixels are wrong.
    wrong1=np.where(s1.data==0, True, False) 
    wrong2=np.where(s2_tr.data==0, True, False)
    wrong=np.logical_or(wrong1, wrong2)
    
    #get transformed pixel spacing
    hs_data=pm_mag.rebin(scale=(rebin_factor,rebin_factor))
    pix_spacing = hs_data.axes_manager.signal_axes[0].scale # nm
    
    phase_images = [pm_mag.data, pm_el.data]
    # phase_images = [im[cropping,cropping] for im in phase_images]
    # wrong = wrong[cropping,cropping]
    processed_phase_images=[]
    for img in phase_images:
        
        #the edge most pixels are wrong, hence erode the edges
        img, wrong = pu.erode_masked_image(img, wrong, radius=erode_depth, wrong_pixel_value=0) 
        
        #fill in holes and other free space with smooth extensions of the edges
        img, temp = pu.fill_masked_image(img, wrong, radius=fill_convolution_radius)
        
        #filter and rebin
        img = skfl.median(img, selem = median_footprint, behavior="ndimage") #remove difference errors and hot pixels
        img = skfl.gaussian(img, sigma=sigma_mag_rec) #smooth to remove high spatial frequency noise
        img = pu.rebin_img(img, rebin_factor=rebin_factor) #reduce pixel number to reduce memory requirements
        img = img.astype(phase_img_dtype)
        processed_phase_images.append(img)
        
    mag_phase, el_phase = processed_phase_images
    
    #adjust confidence array to match
    confidence=np.where(wrong,0,1)
    confidence = pu.rebin_img(confidence, rebin_factor=rebin_factor)
    confidence = np.where(confidence>0.9, 1, 0)
    confidence = confidence.astype(phase_img_dtype)
    
    #save and display
    if save_path is not None:
        Image.fromarray(mag_phase).save(save_path+"_smoothed_rebinned.tif")
    if verbose:
        print("Image shapes:", mag_phase.shape, el_phase.shape, confidence.shape)
        print("Image dtype:", mag_phase.dtype)
        print("Pixel spacing:", pix_spacing, "nm")
    if plot_output:
        pu.matshow_n([mag_phase,el_phase,confidence],["Magnetic phase", "Mean inner potential", "Confidence"])
        
    return(mag_phase, el_phase, confidence, pix_spacing)


def remove_ramp_2d(data_simple, verbose=False, max_iter=200, plot_output=True, lam=0.001, ramp_order=1):
    """
    Run reconstruction_simple on a dataset containing one phasemap and return the phasemap with the ramp subtracted
    """
    data=data_simple
    if verbose:
        print(f"Running CPU reconstruction with {max_iter} iterations and {len(data.mask[data.mask])} voxels")
    magdata_rec, cost_fun = pre.reconstruct_from_phasemaps_simple(data, verbose=verbose, max_iter=max_iter, lam=lam, ramp_order=ramp_order)
    pm=data.phasemaps[0]
    pm_noramp=pm-cost_fun.fwd_model.ramp(0)

    if plot_output:
        magdata_rec.plot_quiver3d(title="M*t")
        pm.plot_phase(title="Mag phase")
        cost_fun.fwd_model.ramp(0).plot_phase(title="ramp")
        pm_noramp.plot_phase(title="Mag phase without ramp")
        amp_2d=np.sum(magdata_rec.field_amp, axis=0)
        plt.matshow(amp_2d)
        plt.title("M*t amp")
    
    return (pm_noramp)
    
def generate_extruded_dataset(phasemap, y_pad=0, plot_input=False, plot_output=False, pad_all_sides=False, data_save_path=None):
    """
    Use one phasemap to generate a CUDA dataset that contains an extruded round 3D mask.
    Takes a 2D mask and generates a 3D mask by rotating it around its centre of mass in y-direction.
    Also creates a projector correspoding to the volume occupied by the 3d mask.

    y_pad=y_extrusion # pad each side of y_axis to allow for boundary regions. In pix

    pad_all_sides: bool (default: False)
        if False, only extends the mask in the +y-direction.
        if True, extends mask in both +y and -y directions.
    
    Returns: (data_set, projector)    

    TODO: centre shift is broken when doing forward projections. workaround implemented
    """

    pm=phasemap.pad(((y_pad,y_pad),(0,0)))
    mask=pm.mask
    
    #define a GPU projector that can generate phaes images from a 3d volume
    projector, mask_sino = generate_projector([mask,])
    r=prt.AstraReconstructor(recon_data=np.ones((1,1,1))) # for plotting
    
    n_proj=mask_sino.shape[1]
    bpr=projector.BP(mask_sino)
    m1=np.where(bpr<n_proj,0,1)
    
    if plot_input: #ispect mask
        r.plot_reconstruction(m1, title="Original Backprojection")
    
    #estimate reduced volume
    xs=np.sum(m1,axis=(0,1))
    indices=np.arange(len(xs))[xs>=1]
    i_left,i_right=np.min(indices),np.max(indices)
    
    ys=np.sum(m1,axis=(0,2))
    indices=np.arange(len(ys))[ys>=1]
    i_top,i_bottom=np.min(indices),np.max(indices)
    
    dimz,dimy,dimx=m1.shape
    dy=dimy//2-(i_bottom+i_top)//2 #centre the edges
    dx=dimx//2-(i_right+i_left)//2
    # shrink_y=((dimy-(2+i_bottom-i_top))//2)*2 #only shrink equally on both sides
    # shrink_x=((dimx-(2+i_right-i_left))//2)*2
    
    shrink_y=np.min([i_top, dimy-1-i_bottom])*2 #no centre shifting
    shrink_x=np.min([i_left, dimx-1-i_right])*2
    
    if plot_input:
        plt.figure()
        plt.plot(xs)
        plt.title("x axis sum")
        plt.xlabel("x, pix")
        plt.ylabel("Sum mask projection")
        
        plt.figure()
        plt.plot(ys)
        plt.title("y axis sum")
        plt.xlabel("y, pix")
        plt.ylabel("Sum mask projection")
    
    
    vol=np.zeros((dimz, dimy-shrink_y+y_pad*2, dimx-shrink_x))
    #corrections to have pixel perfect projection
    if vol.shape[1]%2 != mask_sino.shape[0]%2: #if not both even, make vol y even
        vol=np.pad(vol, ((0,0),(0,1),(0,0)))
    if vol.shape[2]%2 != mask_sino.shape[2]%2: # make vol x even
        vol=np.pad(vol, ((0,0),(0,0),(0,1)))
    centre_shift=(0,0,0)# centre_shift=(-dx,-dy,0) #(x,y,z) broken in forward projections
    projector, mask_sino = generate_projector([mask,], volume=vol, centre_shift=centre_shift)
    
    #generate the 3D mask by rotation
    bpr=projector.BP(mask_sino)
    m2=np.where(bpr<n_proj,0,1)
    m_flat=np.sum(m2,axis=0)
    m_flat=m_flat>0
    m_round=pa.mask_to_3d_round(m_flat,axis=2)
    rdimz,rdimy,rdimx=m_round.shape
    
    #make a projector that fits the 3D mask.
    vol=np.zeros([rdimz,*vol.shape[1:]])
    projector, mask_sino = generate_projector([mask,], volume=vol, centre_shift=centre_shift)

    #adjust mask end by adding extrusions that may be assigned boundary values later
    if y_pad > 0:
        m_top=np.transpose(np.tile(m_round[:,-(y_pad+3),:], (y_pad,1,1)), axes=(1,0,2)) #avoid copying empty space
        m_round[:,-(y_pad):,:]=m_top #'top' means large y values
        m_round[:,-(y_pad),:]=0 # include a break
        if pad_all_sides:
            m_end=np.transpose(np.tile(m_round[:,y_pad+2,:], (y_pad,1,1)), axes=(1,0,2))
            m_round[:,:y_pad,:]=m_end
            m_round[:,y_pad,:]=0
    
    # compare backprojection and rotated mask
    bpr=projector.BP(mask_sino)
    m3=np.where(bpr<n_proj,0,1)
    if m3.shape!=m_round.shape:
        raise ValueError("Projected and rotated mask dimensions do not match")
    if plot_input:
        r.plot_reconstruction(m3, title="Backprojected mask")
    if plot_input:
        r.plot_reconstruction(m_round, title="3D extruded mask")

    # define dataset to be used for reconstruction
    pix_size=pm.a
    dim=m_round.shape
    data_set=DataSetCUDA(pix_size, dim, mask=m_round)
    dummy_projector=DummyProjector(dim=dim, dim_uv=pm.dim_uv)
    for i in range(mask_sino.shape[1]): #do once for each projection
        data_set.append(pm,dummy_projector)
    data_set.projector_params=(projector.pg, projector.vg)

    if plot_output:
        data_set.plot_mask(pretty=True, title="3D extruded mask")
        data_set.phasemaps[0].plot_phase()
    
    if data_save_path is not None:
        with open(data_save_path, "wb") as f:
            pickle.dump(data_set,f)
        print("Saving as",data_save_path)

    return(data_set, projector)
    
    
def get_cost_function_uniform_ms(data_set, projector, y_cutoff=0, M_axis=1, mask_threshold=None, gauss_sigma=0):
    """
    Returns a functions that gives the sum absolute difference between experimental and simulated phase map.
    Simulations is perfomed by assuming the 3d mask is uniformly magnetised.

    Returns: cost_for_ms
        def cost_for_ms(x, data_set=data_set, projector=projector, y_cutoff=y_cutoff,
                    plot_result=False, print_cost=False, M_axis=M_axis):
    """
    def cost_for_ms(x, data_set=data_set, projector=projector, y_cutoff=y_cutoff,
                    plot_result=False, print_cost=False, M_axis=M_axis):
        ms,c=x
        m_guess=np.zeros((3,*data_set.dim))
        m_guess[M_axis,data_set.mask]=ms
        m_guess=pr.VectorData(data_set.a, m_guess)
        pms=data_set.create_phasemaps(m_guess, projector=projector)
        pm=pms[0]
        real=data_set.phasemaps[0].phase[y_cutoff:-y_cutoff,:]
        conf=data_set.phasemaps[0].confidence[y_cutoff:-y_cutoff,:]
        conf_mask=conf<1
        real=skfl.gaussian(real, sigma=gauss_sigma)
        sim=pm.phase[y_cutoff:-y_cutoff,:]+c
        sim=skfl.gaussian(sim, sigma=gauss_sigma)
        real[conf_mask]=0
        sim[conf_mask]=0
        phase_diff=real-sim
        cost=np.sum(np.abs(phase_diff))#
        if print_cost:
            print("sum abs diff", cost)
        if plot_result:
            vmin,vmax=np.min(real),np.max(real)
            pu.subplots_n([phase_diff,real,sim],labels=["diff","real","sim"], 
                          vmax=vmax, vmin=vmin, title=f"{mask_threshold}. Rad, [{vmin:.2f},{vmax:.2f}]")
            pm.plot_phase(vmin=vmin,vmax=vmax)
            data_set.phasemaps[0].plot_phase(vmin=vmin,vmax=vmax)
        return(cost)
    return (cost_for_ms)

def get_cost_function_two_ms(data_set, projector, y_cutoff=0, M_axis=1, mask_threshold=None, gauss_sigma=0, material_y_interface=None):
    """
    Returns a functions that gives the sum absolute difference between experimental and simulated phase map.
    Simulations is perfomed by assuming the 3d mask is uniformly magnetised.

    Returns: cost_for_ms
        def cost_for_ms(x, data_set=data_set, projector=projector, y_cutoff=y_cutoff,
                    plot_result=False, print_cost=False, M_axis=M_axis):
    """
    # if material_y_interface is None:
    #     material_y_interface=data_set.phasemaps[0].dim_uv[0]//2
        
    def cost_for_ms(x, data_set=data_set, projector=projector, y_cutoff=y_cutoff,
                    plot_result=False, print_cost=False, M_axis=M_axis, gauss_sigma=gauss_sigma):
        ms1,ms2,material_y_interface,c=x
        m_guess=np.zeros((3,*data_set.dim))
        m_guess[M_axis,data_set.mask]=ms1
        half_mask=data_set.mask.copy()
        half_mask[:,int(material_y_interface):,:]=False
        m_guess[M_axis,half_mask]=ms2
        m_guess=pr.VectorData(data_set.a, m_guess)
        pms=data_set.create_phasemaps(m_guess, projector=projector)
        pm=pms[0]
        real=data_set.phasemaps[0].phase[y_cutoff:-y_cutoff,:]
        conf=data_set.phasemaps[0].confidence[y_cutoff:-y_cutoff,:]
        conf_mask=conf<1
        real=skfl.gaussian(real, sigma=gauss_sigma)
        sim=pm.phase[y_cutoff:-y_cutoff,:]+c
        sim=skfl.gaussian(sim, sigma=gauss_sigma)
        real[conf_mask]=0
        sim[conf_mask]=0
        phase_diff=real-sim
        cost=np.sum(np.abs(phase_diff))#
        if print_cost:
            print("sum abs diff", cost)
        if plot_result:
            vmin,vmax=np.min(real),np.max(real)
            pu.subplots_n([phase_diff,real,sim],labels=["diff","real","sim"], 
                          vmax=vmax, vmin=vmin, title=f"{mask_threshold}. Rad, [{vmin:.2f},{vmax:.2f}]")
            pm.plot_phase(vmin=vmin,vmax=vmax)
            data_set.phasemaps[0].plot_phase(vmin=vmin,vmax=vmax)
        return(cost)
    return (cost_for_ms)

def get_cost_function_two_ms_rigid(data_set, projector, y_cutoff=0, M_axis=1, mask_threshold=None, gauss_sigma=0, material_y_interface=None):
    """
    material_y_interface does not move.
    
    Returns a functions that gives the sum absolute difference between experimental and simulated phase map.
    Simulations is perfomed by assuming the 3d mask is uniformly magnetised.

    Returns: cost_for_ms
        def cost_for_ms(x, data_set=data_set, projector=projector, y_cutoff=y_cutoff,
                    plot_result=False, print_cost=False, M_axis=M_axis):
    """
    # if material_y_interface is None:
    #     material_y_interface=data_set.phasemaps[0].dim_uv[0]//2
        
    def cost_for_ms(x, data_set=data_set, projector=projector, y_cutoff=y_cutoff,
                    plot_result=False, print_cost=False, M_axis=M_axis):
        ms1,ms2,c=x
        m_guess=np.zeros((3,*data_set.dim))
        m_guess[M_axis,data_set.mask]=ms1
        half_mask=data_set.mask.copy()
        half_mask[:,int(material_y_interface):,:]=False
        m_guess[M_axis,half_mask]=ms2
        m_guess=pr.VectorData(data_set.a, m_guess)
        pms=data_set.create_phasemaps(m_guess, projector=projector)
        pm=pms[0]
        real=data_set.phasemaps[0].phase[y_cutoff:-y_cutoff,:]
        conf=data_set.phasemaps[0].confidence[y_cutoff:-y_cutoff,:]
        conf_mask=conf<1
        real=skfl.gaussian(real, sigma=gauss_sigma)
        sim=pm.phase[y_cutoff:-y_cutoff,:]+c
        sim=skfl.gaussian(sim, sigma=gauss_sigma)
        real[conf_mask]=0
        sim[conf_mask]=0
        phase_diff=real-sim
        cost=np.sum(np.abs(phase_diff))#
        if print_cost:
            print("sum abs diff", cost)
        if plot_result:
            vmin,vmax=np.min(real),np.max(real)
            pu.subplots_n([phase_diff,real,sim],labels=["diff","real","sim"], 
                          vmax=vmax, vmin=vmin, title=f"{mask_threshold}. Rad, [{vmin:.2f},{vmax:.2f}]")
            pm.plot_phase(vmin=vmin,vmax=vmax)
            data_set.phasemaps[0].plot_phase(vmin=vmin,vmax=vmax)
        return(cost)
    return (cost_for_ms)
    
    
def get_cost_function_field_ms(data_set, projector, mfield, y_cutoff=0, M_axis=1, mask_threshold=None, gauss_sigma=0):
    """
    TODO: why need mask_threshold? mask_threshold is only for the title -> rename in both versions.
    
    Returns a functions that gives the sum absolute difference between experimental and simulated phase map.
    Simulation is perfomed by assuming the 3d mask contains a predefined distribution 'mfield'.

    mfield: pyramid VectorData object
    projector: OpTomo projector
    data_set: mbir-tem DataSetCUDA

    
    Returns: cost_for_ms
        def cost_for_ms(x, data_set=data_set, projector=projector, y_cutoff=y_cutoff,
                    plot_result=False, print_cost=False, M_axis=M_axis, gauss_sigma=gauss_sigma):
    """
    ms_max=np.max(mfield.field_amp)
    mfield.field=mfield.field/ms_max
    
    def cost_for_ms(x, data_set=data_set, projector=projector, mfield=mfield, y_cutoff=y_cutoff,
                    plot_result=False, print_cost=False, M_axis=M_axis, gauss_sigma=gauss_sigma):
        ms,c=x
        m_guess=mfield.copy()
        m_guess.field=m_guess.field*ms
        m_guess.a=data_set.a
        pms=data_set.create_phasemaps(m_guess, projector=projector)
        pm=pms[0]
        real=data_set.phasemaps[0].phase[y_cutoff:-y_cutoff,:]
        real=skfl.gaussian(real, sigma=gauss_sigma)
        sim=pm.phase[y_cutoff:-y_cutoff,:]+c
        sim=skfl.gaussian(sim, sigma=gauss_sigma)
        phase_diff=real-sim
        cost=np.sum(np.abs(phase_diff))#
        if print_cost:
            print("sum abs diff", cost)
        if plot_result:
            vmin,vmax=np.min(real),np.max(real)
            pu.subplots_n([phase_diff,real,sim],labels=["diff","real","sim"], 
                          vmax=vmax, vmin=vmin, title=f"{mask_threshold}. Rad, [{vmin:.2f},{vmax:.2f}]")
            pm.plot_phase(vmin=vmin,vmax=vmax)
            data_set.phasemaps[0].plot_phase(vmin=vmin,vmax=vmax)
        return(cost)
    return (cost_for_ms)