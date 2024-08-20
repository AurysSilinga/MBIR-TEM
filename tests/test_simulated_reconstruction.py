# -*- coding: utf-8 -*-
"""Testcase for the simplified reconstruction process."""

#import correctly
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pyramid as pr
import mbir as prs
import hyperspy.api as hs
import skimage.filters as skfl
import fpd

import mayavi
from mayavi import mlab

from numpy.testing import assert_allclose


class TestCaseReconstruction(unittest.TestCase):
    """TestCase for the analytic module."""
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_simulated_reconstruction')
    
    @mlab.show
    def test_reconstruction(self):
        """Test of the reconstruction. Simulates a magnetic object, calculates phasemaps, 
        and then reconstructs. Also checks for truthfullness along the way"""
        #create a simulated object

        r=6 #radius in pixels
        height=5
        a_pix=1 # nm pixel size
        axis='x'
        dim=(r*2+2,r*2+2,height)

        mag_vor=pr.magcreator.examples.vortex_disc(a=a_pix, dim=dim, radius=r, height=height, axis=axis)

        phi=np.radians(0) #azimuthal angle (from x-axis) for magnetisation
        theta=np.radians(90) #polar angle (from z axis)

        mag_rod=pr.magcreator.examples.homog_disc(a=a_pix,dim=dim,radius=r+1,height=height,phi=phi,theta=theta,axis=axis)

        fields=[mag_vor.field, mag_rod.field]
        mag_needle=np.concatenate(fields, axis=-1)
        mag_needle = pr.VectorData(a_pix, mag_needle)
        mag_needle.dim

        mask=mag_needle.get_mask()
        wrongs = np.bitwise_not(mask)

        components=[]
        for i in range(3):
            component=mag_needle.field[i,...]
            component, remaining_wrongs=prs.alignment.fill_masked_image(component, wrongs, radius=1)
            smoothed_component=skfl.gaussian(component, sigma=3)
            smoothed_component[wrongs]=0
            components.append(smoothed_component)
            
            # creating a vector field and filling free space preserves dimensions and fills completely
            assert(smoothed_component.shape == (14,14,10))
            assert(np.all(remaining_wrongs==np.full(remaining_wrongs.shape, False)))
            
        mag_needle.field=np.array(components)

        mag_needle.plot_quiver3d(ar_dens=2)
        mag_needle.plot_mask()
        mag_needle.dim
        
        # test is plotting executes and returns a non-zero object

        c=mag_needle.plot_quiver3d()
        b=mag_needle.plot_mask()
        a=mag_needle.plot_quiver_field()

        assert(type(c) is mayavi.modules.vectors.Vectors)
        assert(type(b) is mayavi.modules.iso_surface.IsoSurface)

        fig=plt.Figure()
        axis = fig.add_subplot(1, 1, 1)

        assert(type(a) == type(axis))
        
        #Make a dataset

        mag_field=mag_needle.copy()
        b_s=0.3 #reduce magnetisation to see if division errors appear.
        x_angs=np.arange(-90, 60+1, 30) #deg
        z_angs=(0, 90)
        camera_rotation=0 #deg
        dim_uv = (16,16) # CHANGE THIS TO BREAK EVERYTHING

        x_angs=np.arange(-90, 60+1, 30) #deg
        z_angs=(0, 90)

        projection_x_ang = np.concatenate([x_angs]*len(z_angs), axis=-1)

        projection_z_ang = [] 
        for i in range(len(z_angs)):
            projection_z_ang = projection_z_ang +  [z_angs[i]]*len(x_angs)

        data=prs.simulation.make_phasemap_dataset (projection_x_ang, projection_z_ang, mag_field, b_s, camera_rotation=camera_rotation, center=None,
                                   subcount=1, b_unit=1, mask_threshold=0, mask_overlap_threshold=1, plot_results=False, dim_uv=dim_uv)

        data.set_3d_mask()
        assert(np.all(mag_needle.get_mask() == data.mask)) #check if mask is preserved
        
        #check if phasemaps can be shown

        data.plot_phasemaps_combined()
        
        #test equivalent projection angles

        mask3d=data.mask.astype(int)
        p0 = prs.alignment.project_scalar_array(mask3d, zrot=0, xrot=0, subcount=1, dim_uv=dim_uv)
        p1 = np.rot90(prs.alignment.project_scalar_array(mask3d, zrot=90, xrot=0, subcount=1, dim_uv=dim_uv), k=1)
        assert(np.all(p0 == p1))

        p2 = np.rot90(prs.alignment.project_scalar_array(mask3d, zrot=90, xrot=0, crot=90 ,subcount=1, dim_uv=dim_uv), k=2)
        assert(np.all(p0 == p2))

        p3 = prs.alignment.project_scalar_array(mask3d, zrot=180, xrot=90, crot=180 ,subcount=1, dim_uv=dim_uv)
        assert(np.all(p0 == p3))
        
        # perform the reconstruction

        regulariser_strengths = 1e-7
        mag_guess = None
        iterations=1000
        sinle_run=1000 #iterations
        result_note=f"{iterations} iterations for unit testing"
        cost_values=[]

        runs=iterations//sinle_run
        magdata_rec=mag_guess
        for i in range(runs):

            magdata_rec, cost_function = prs.reconstruction.reconstruct_from_phasemaps(data, lam=regulariser_strengths, 
                                                max_iter=sinle_run, ramp_order=1, verbose=False, plot_input=False, plot_results=False, 
                                               regulariser_type='exchange',mag_0=magdata_rec)
            prs.reconstruction.append_valid_costfunction_values(cost_values, cost_function)
            
        result = (magdata_rec, cost_function, cost_values, result_note)
        
        magdata_rec, cost_function, cost_values, result_note = result

        # if mask shape is preserved
        assert np.all(magdata_rec.get_mask()==mag_needle.get_mask()) 

        # if subtraction and plotting still works
        test= magdata_rec-mag_needle*b_s
        magdata_rec.plot_quiver3d()
        test.plot_quiver3d(coloring='amplitude')
        print("sum residuals between truth and reconstructed", np.sum(test.field))

        #if close to the original magnetisation
        assert_allclose(magdata_rec.field, mag_needle.field, atol=np.max(np.abs(mag_needle.field)), rtol=0, 
                        err_msg="The reconstruction is very different from the original magnetisation", verbose=False)

        magdata_rec_orig = np.load(os.path.join(self.path, "reconstructed_magnetisation.npy"))

        #if close to previous reconstruction
        assert_allclose(magdata_rec.field, magdata_rec_orig, atol=0.05*np.max(np.abs(magdata_rec_orig)), rtol=0, 
                        err_msg="The reconstruction is more than 5% different from previous reconstructions", verbose=False)
        
        # remove all unwanted interactive windows
        mayavi.mlab.close(all=True)
        plt.close('all')
