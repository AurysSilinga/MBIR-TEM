# -*- coding: utf-8 -*-
# Aurys Silinga, 2024
#

"""
Simulation functions for pyramid(by)AS
"""

import pyramid as pr
import matplotlib.pyplot as plt
import numpy as np
from .util import *


def simulate_reconstruction(data, magdata_rec, cost, b_0=1, unit='rad', plot_results=True, mask_threshold=0, **kwargs):
    """
    Compare original data and that that would be obtained from reconstructed vector field 'magdata_rec'.
    """
    phases=np.concatenate([np.ravel(pm.phase) for pm in data.phasemaps])
    vmax = np.nanmax(phases)
    vmin = np.nanmin(phases)
    phasemaps_rec=[]
    
    for i in range(len(data.phasemaps)):
        phasemap = data.phasemaps[i]
        projector = data.projectors[i]

        ramp = cost.fwd_model.ramp(i)
        mag_projection=projector(magdata_rec) #project magnetic field
        
        #calculate phase map
        mapper_kernel=pr.Kernel(data.a, phasemap.dim_uv, b_0=b_0) #define phase calculator parameters
        phas_mapper=pr.PhaseMapperRDFC(mapper_kernel) #create a phase calculator
        phasemap_r=phas_mapper(mag_projection)
        
        #project magnetic field to obtain mask
        amp_field = pr.ScalarData(magdata_rec.a, magdata_rec.field_amp) 
        amp_projection=projector(amp_field) 
        mask= amp_projection.get_mask(threshold=mask_threshold)
        phasemap_r.mask=mask[0,:,:]         
        
        #add ramp effects
        phasemap_r = phasemap_r + ramp
        phasemaps_rec.append(phasemap_r)
        
        if plot_results:
            phasemap.plot_phase(title="Original: " + projector.get_info(), vmin=vmin, vmax=vmax, 
                                unit=unit, show_conf=False, **kwargs)
            phasemap_r.plot_phase(title="Reconstructed: " + projector.get_info(), vmin=vmin, vmax=vmax, 
                                  unit=unit, show_conf=False, **kwargs)
            diff = phasemap-phasemap_r
            diff.phase[phasemap.confidence < 1]=0
            diff.plot_phase(title="Difference Orig - Rec: " + projector.get_info(), vmin=vmin, vmax=vmax, 
                            unit=unit, show_conf=False, **kwargs)
        
    return (phasemaps_rec)


def make_phasemap_dataset (projection_x_ang, projection_z_ang, mag_field, b_s, camera_rotation=0, center=None,
                           subcount=5, b_unit=1, mask_threshold=0, mask_overlap_threshold=1, plot_results=False, dim_uv=None):
    """
    Takes a set of angles in degrees and a B field 3D vector field and returns a set of phasemaps and projector objects
    Creates an artificial CT phasemap dataset for testing reconstruction.
    
    projection_x_ang=[0,90,45] #deg
    projection_z_ang=[0,0,45]
    mag_field=pr.Vector field
    subcount=11 #workaround for 3d -> 2d interpolation by checking every subpixel for neighbours that are close enough.
    # Odd numbers for pixels to be centered. Causes blurring if too high.
    R=0.5 #magical equivalence radius for finding relevant pixels around a point
    dim=mag_field.dim
    a_spacing=mag_field.a
    b_unit=1 #use units of 1 T
    b_s = 3 #the saturation magnetisation of the model, in units of b_unit. 
        Magnetisation = b_s * mag_field / b_unit
    mask_threshold=0 #field amplitude > thresholf is included in the mask
    mask_overlap_threshold=0.9 #if calculating 3D mask from multiple 2D masks, point is included if >90% of masks overlap there
    
    
    """
    dim=mag_field.dim
    a_spacing=mag_field.a
    RP_Projector=pr.projector.RotTiltProjector
    data = pr.DataSet(a_spacing, dim, b_0=b_unit) #initialise empty dataset of phase maps
    projection_x_ang=np.radians(projection_x_ang)
    projection_z_ang=np.radians(projection_z_ang)
    camera_rotation = np.radians(camera_rotation)
    
    if dim_uv==None:
        dim_uv=(np.max(dim), np.max(dim))
    
    mapper_kernel=pr.Kernel(a_spacing, dim_uv, b_0=b_s) #define phase calculator parameters
    phas_mapper=pr.PhaseMapperRDFC(mapper_kernel) #create a phase calculator

    #inside for loop
    for i in range(len(projection_x_ang)):
        x_ang=projection_x_ang[i]
        z_ang=projection_z_ang[i]
        
        #define the projection
        projector = RP_Projector(dim, z_ang, x_ang, camera_rotation=camera_rotation, center=center, subcount=subcount, dim_uv=dim_uv)
        mag_projection=projector(mag_field) #project magnetic field

        #create phase map
        amp_field = pr.ScalarData(mag_field.a, mag_field.field_amp)
        amp_projection=projector(amp_field) #project magnetic field
        mask= amp_projection.get_mask(threshold=mask_threshold) 
        
        phasemap=phas_mapper(mag_projection)
        phasemap.mask=mask[0,:,:] #set individual mask

        data.append(phasemap, projector)
        print(i+1,"/",len(projection_x_ang),end='; ')

    #calculate mask
    data.set_3d_mask(threshold=mask_overlap_threshold)
    data.set_Se_inv_diag_with_conf()
    
    if plot_results:
        data.plot_mask()
        data.plot_phasemaps_combined()
    
    #check if done correctly
    print("data dimensions",data.dim)
    return(data)


