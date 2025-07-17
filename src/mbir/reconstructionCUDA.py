# -*- coding: utf-8 -*-
# Aurys Silinga, 2025
#

"""
Functions for CUDA accelerated MBIR reconstruction
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
import cupy as cp
import cupyx.scipy.fft as cpfft
from timeit import default_timer as timer


def generate_projector(images, volume=None, projection_x_ang=0, centre_shift=(0,0,0), projection_z_ang=0, camera_rotation=0, verbose=False):
    """
    Uses a set of images and angles to generate an OpTomo pojector and a sinogram.
    
    If images are given, the 3d volume will be determined automatically if set to None. 
    Custom volume sizes can be set by passing a numpy array of preffered size

    images: list 
        [im1,im2, ...,imN]. the projections. Sets projection size and defines the dimensions of the sinogram.
    volume: 3d np.array
        The volume that is populated by reconstruction. Sets the size of the reconstruction volume.
    projection_z_ang: 0
        sample rotation relative to tilt axis. in deg
    projection_x_ang: 0
        sample tilt
    camera_rotation: 0
        camera_rotation
    centre_shift: (0,0,0)
        (dx, dy, dz). Moves the centre of the 3D volume.

    returns: (projector, sinogram)

    TODO: centre shift is broken when doing forward projections. 0 works fine.
    """

    mask_sino = np.transpose([im for im in images], axes=[1,0,2])[::-1,:,:] # correction for weird coordinate transforms
    if isinstance(projection_x_ang, (float, int)):
        projection_x_ang=[projection_x_ang,]*mask_sino.shape[1] #if multiple images at constant tilt are given
    r=prt.AstraReconstructor(mask_sino, volume, projection_x_ang=projection_x_ang, projection_z_ang=projection_z_ang, 
                             camera_rotation=camera_rotation, verbose=verbose)
    r.move_reconstruction_centre(centre_shift)
    proj_geom=r.proj_geom
    vol_geom=r.vol_geom
    # hack for single frame images
    if proj_geom['Vectors'].shape[0]==1:
        proj_geom['Vectors']=np.tile(proj_geom['Vectors'], (2,1))
        mask_sino=np.tile(mask_sino, (1,2,1))
    proj_id=astra.create_projector('cuda3d', proj_geom, vol_geom)
    projector = astra.OpTomo(proj_id)
    astra.projector3d.delete(proj_id)

    return(projector, mask_sino)
    



#reconstruction

class DummyProjector(pr.projector.Projector):
    """Class representing a unused placeholder projector. Is only included to store metadata.

    Attributes
    ----------
    dim : tuple (N=3)
        Dimensions (z, y, x) of the magnetization distribution.
    dim_uv : tuple (N=2)
        Dimensions (v, u) of the projected grid.
    rotation : float
        Angle in `rad` describing the rotation around the z-axis before the tilt is happening.
    tilt : float
        Angle in `rad` describing the tilt of the beam direction relative to the x-axis.
    camera_rotation : float (optional)
        Angle in `rad` describing the rotation around the z-axis before after the tilt.

    """

    def __init__(self, dim=None, dim_uv=None, tilt=0, rotation=0, camera_rotation=0, weight = np.ones((1,1)), coeff = np.ones((1,1))):
        self.tilt=tilt
        self.rotation=rotation
        self.camera_rotation=camera_rotation
        super().__init__(dim, dim_uv, weight, coeff)

    def jac_dot(self, vector):
        raise NotImplementedError("""Single image projection is not implemented. Use Astra.OpTomo and CUDA forward model 
        to project all phase images simultaneously.""")
    def jac_T_dot(self, vector):
        raise NotImplementedError("""Single image projection is not implemented. Use Astra.OpTomo and CUDA forward model 
        to project all phase images simultaneously.""")
        
    def get_info(self, verbose=False):
        """Get specific information about the projector as a string.

        Parameters
        ----------
        verbose: boolean, optional
            Overwritten, not used.

        Returns
        -------
        info : string
            Information about the projector as a string, e.g. for the use in plot titles.

        """
        theta_ang = self.rotation * 180 / np.pi
        phi_ang = self.tilt * 180 / np.pi
        return R'z-rot={:.1f}°, x-tilt={:.1f}°'.format(theta_ang, phi_ang)

class DataSetCUDA(pr.dataset.DataSet):
    """
    TODO: attach forward model to the dataset, instead of just projector.
    
    Class for collecting phase maps and corresponding projectors.

    Represents a collection of (e.g. experimentally derived) phase maps, stored as
    :class:`~.PhaseMap` objects and corresponding projectors stored as :class:`~.Projector`
    objects. At creation, the grid spacing `a` and the dimension `dim` of the magnetization
    distribution have to be given. Data can be added via the :func:`~.append` method, where
    a :class:`~.PhaseMap`, a :class:`~.Projector` and additional info have to be given.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    dim: tuple (N=3)
        Dimensions of the 3D magnetization distribution.
    b_0: float
        Magnetic induction units in `T`.
    mask: :class:`~numpy.ndarray` (N=3), optional
        A boolean mask which defines the magnetized volume in 3D.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `NxN` with N
        being the length of the targetvector y (vectorized phase map information).
    projectors: list of :class:`~.Projector`
        A list of all stored phasemap metadata regarding orientation
    phasemaps: list of :class:`~.PhaseMap`
        A list of all stored :class:`~.PhaseMap` objects.
    projector_params: tuple of dict, optional
        (projector.pg, projector.vg) astra.OpTomo projector that simultaneously projects all phasemaps.
    """
    
    def __init__(self, a, dim, b_0=1, mask=None, Se_inv=None, projector=None):
        super().__init__(a, dim, b_0, mask, Se_inv)
        if projector is not None:
            self.projector_params=(projector.pg, projector.vg)
            
    def get_projector(self):
        proj_geom, vol_geom=self.projector_params
        proj_id=astra.create_projector('cuda3d', proj_geom, vol_geom)
        projector = astra.OpTomo(proj_id)
        astra.projector3d.delete(proj_id)
        return(projector)

    def set_3d_mask(self, projector=None, mask_list=None, threshold=1.0):
        # should be called backproject_3d_mask
        """
        Set the 3D mask from a list of 2D masks by direct backprojection. 
        Only Used for diagnostics. Precise mask creation is perfomed during alignment instead of here.

        Parameters
        ----------
        projector: OpTomo projector (optional)
            Will be generated based on stored geometries if set to None.
        mask_list: list of :class:`~numpy.ndarray` (optional)
            List of 2D masks, which represent the projections of the 3D mask. If not given this
            uses the mask matrizes of the phase maps. If just one phase map is present, the
            according mask is simply expanded to 3D and used directly.
        threshold: float, optional
            The threshold, describing the minimal number of 2D masks which have to extrude to the
            point in 3D to be considered valid as containing magnetisation. `threshold` is a
            relative number in the range of [0, 1]. The default is 1.0. Choosing a value of 1 is
            the strictest possible setting (every 2D mask has to contain a 3D point to be valid).

        Returns
        -------
            None

        """
        if projector is None:
            proj_geom, vol_geom=self.projector_params
            proj_id=astra.create_projector('cuda3d', proj_geom, vol_geom)
            projector = astra.OpTomo(proj_id)
            astra.projector3d.delete(proj_id)
        
        self._log.debug('Calling set_3d_mask')
        if mask_list is None:  # if no masks are given, extract from phase maps:
            mask_list = [phasemap.mask for phasemap in self.phasemaps]
        #backproject
        mask_sino = np.transpose(mask_list, axes=[1,0,2])[::-1,:,:] #to agree with pyramid coordinates
        mask_3d=projector.BP(mask_sino)
        #threshold
        mask_3d=(mask_3d>=threshold*self.count)
        #set
        self.mask = mask_3d

    def create_phasemaps(self, magdata, difference=False, ramp=None, projector=None):
        """Create a list of phasemaps with the projectors in the dataset for a given
        :class:`~.VectorData` object.

        Parameters
        ----------
        magdata : :class:`~.VectorData`
            Magnetic distribution to which the projectors of the dataset should be applied.
        difference : bool, optional
            If `True`, the phasemaps of the dataset are subtracted from the created ones to view
            difference images. Default is False.
        ramp : :class:`~.Ramp`
            A ramp object, which can be specified to add a ramp to the generated phasemaps. 
            Found at 'cost_function.fwd_model.ramp'. 
            if ramp==None and difference == True, the images show reconstructions errors+ramp.

        Returns
        -------
        phasemaps : list of :class:`~.phasemap.PhaseMap`
            A list of the phase maps resulting from the projections specified in the dataset.

        """
        self._log.debug('Calling create_phasemaps')

        #define input parameters and forward model
        vfield=magdata
        temp_mask=self.mask.copy() 
        self.mask=vfield.get_mask() #to allow different numbers of vectors in magdata.
        if projector is None:
            proj_geom, vol_geom=self.projector_params
            proj_id=astra.create_projector('cuda3d', proj_geom, vol_geom)
            projector = astra.OpTomo(proj_id)
            astra.projector3d.delete(proj_id)
        fwd_model=ForwardModelCUDA(self, projector, ramp_order=None)
        dim_uv=self.phasemaps[0].dim_uv
        phasemaps=self.phasemaps
        n_proj=len(phasemaps)
    
        # use forward model to generate the phasemaps
        phasemaps_rec=[]
        masks=projector.FP(vfield.get_mask())
        masks=np.transpose(masks, axes=[1,0,2])[:,::-1,:] # transpose such that first axis is tilt angle #pyramid coordinate corr
        masks=(masks>0.6) #pixel is accepted if it is mostly filled
        confidences=np.ones((n_proj,)+dim_uv)
        phases=fwd_model.vector_to_phase( fwd_model( fwd_model.vfield_to_vector(vfield)))

        for i in range(n_proj):
            pm=pr.PhaseMap(vfield.a, phases[i,:,:], mask=masks[i,:,:], confidence=confidences[i,:,:])
            if ramp is not None:
                pm += ramp(i)
            if difference:
                pm-=phasemaps[i]
                pm.phase[phasemaps[i].confidence<1]=0

            phasemaps_rec.append(pm)
        self.mask=temp_mask #return to previous state
        return(phasemaps_rec)




class ForwardModelCUDA(pr.ForwardModel):
    """
    Base pyramid Forward model accelerated with cuda.
    Does not need inbuilt projectors, but instead uses an atra OpTomo projector for all projections.
    self.projector (astra.optomo.OpTomo) needs to be set externally.
    Only allows one phasemapper kernel to be used and all images need to be the same size.

    data_set: :class:`~dataset.DataSet`
        :class:`~dataset.DataSet` object, which stores all required information for calculation.
    projector: astra.optomo
        A projector object optimised for acceleration with CUDA. 
        Projects 3D volume into a sinogram stack in one call, which is optimised for GPU.
    ramp_order : int or None (default)
        Polynomial order of the additional phase ramp which will be added to the phase maps.
        All ramp parameters have to be at the end of the input vector and are split automatically.
        Default is None (no ramps are added)
    """

    def __init__(self, data_set, projector, ramp_order=None):
        """
        Represents a strategy for the mapping of a 3D magnetic distribution to two-dimensional
        phase maps. A :class:`~.DataSet` object is given which is used as input for the model
        (projectors, phasemappers, etc.). A `ramp_order` can be specified to add polynomial ramps
        to the constructed phase maps (which can also be reconstructed!). A :class:`~.Ramp` class
        object will be constructed accordingly, which also holds all info about the ramps after a
        reconstruction.
        
        data_set: :class:`~dataset.DataSet`
            :class:`~dataset.DataSet` object, which stores all required information for calculation.
        projector: astra.optomo
            A projector object optimised for acceleration with CUDA. 
            Projects 3D volume into a sinogram stack in one call, which is optimised for GPU.
        ramp_order : int or None (default)
            Polynomial order of the additional phase ramp which will be added to the phase maps.
            All ramp parameters have to be at the end of the input vector and are split automatically.
            Default is None (no ramps are added)
        
        """
        super().__init__(data_set, ramp_order=ramp_order)
        self.projector=projector
    
    def fast_jac(self, mag_vector, use_vector=True, time_execution=False):
        t0=timer()
        if time_execution: print(timer()-t0, 's, starting')
        mag_field=self.magdata
        projector=self.projector
        
        if use_vector:
            Mx,My,Mz = np.reshape(mag_vector, mag_field.field.shape)
        else:
            Mx,My,Mz = mag_field.field
        
        M=(Mz,My,Mx)
        geom_vectors=projector.pg['Vectors']
        det_vec_y=(geom_vectors[:,9:12]) #y
        det_vec_x=(geom_vectors[:,6:9]) #x
        n_proj=geom_vectors.shape[0]
        dim_uv=(projector.pg['DetectorRowCount'],projector.pg['DetectorColCount']) #used to define kerner.dim_uv
        
        axes=((0,0,1),(0,1,0),(1,0,0)) # (z ,y ,x)
        M_u=np.zeros((dim_uv[0],n_proj,dim_uv[1])) #does not allow different size kernels.
        M_v=np.zeros((dim_uv[0],n_proj,dim_uv[1]))

        if time_execution: print(timer()-t0, 's, setup done')
        for i in range(len(axes)):
            Mn=M[i]
            axis=axes[i]
            
            #vector component projection onto detector plane
            coef_u = np.dot(det_vec_y,axis) 
            coef_v = np.dot(det_vec_x,axis)

            #pyramid weirdness
            coef_u,coef_v=coef_v,coef_u
            if i==1:
                coef_u=-1*coef_u
                coef_v=-1*coef_v
            
        
            #volume projection onto detector plane
            M_mag = projector.FP(Mn[:,::-1,:]) # shape = (dim_u, n_proj, dim_v) #does the same thing thing as astra.create_sino3d_gpu, but a bit faster.
            M_u+=M_mag*coef_u[np.newaxis,:,np.newaxis]
            M_v+=M_mag*coef_v[np.newaxis,:,np.newaxis]

        if time_execution: print(timer()-t0, 's, projection done')
            
        kernel = pr.Kernel(mag_field.a, dim_uv)
        
        m_u = cp.transpose(M_u, axes=(1,0,2))  # u-component
        m_v = cp.transpose(M_v, axes=(1,0,2))  # v-component
    
        #extent kernel to process all phasemaps in parallel way
        padded_dim=(n_proj,)+kernel.dim_pad
        padded_mag_slice=(slice(None),)+kernel.slice_mag
        padded_phase_slice=(slice(None),)+kernel.slice_phase
    
        # pad with empty space (double the size) to reduce aliasing artefacts
        u_mag = cp.zeros(padded_dim, dtype=kernel.u.dtype)
        v_mag = cp.zeros(padded_dim, dtype=kernel.u.dtype)
        u_mag[padded_mag_slice] = cp.array(m_u)  # u-component
        v_mag[padded_mag_slice] = cp.array(m_v)  # v-component
        
        # perform convolution
        u_mag_fft = cpfft.rfft2(u_mag)
        v_mag_fft = cpfft.rfft2(v_mag)
        phase_fft = u_mag_fft * cp.array(kernel.u_fft) + v_mag_fft * cp.array(kernel.v_fft)
        phase = cpfft.irfft2(phase_fft)[padded_phase_slice]

        if time_execution: print(timer()-t0, 's, convolution done')
        
        return(cp.asnumpy(phase))

    def jac_dot(self, x, vector):
        """Calculate the product of the Jacobi matrix with a given `vector`.
        Sped up version using CUDA graphics card acceleration.
        Uses self.fast_jac
    
        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Evaluation point of the jacobi-matrix. The Jacobi matrix is constant for a linear
            problem, thus `x` can be set to None (it is not used int the computation). It is
            implemented for the case that in the future nonlinear problems have to be solved.
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the 3D magnetization distribution. First the `x`, then the `y` and
            lastly the `z` components are listed. Ramp parameters are also added at the end if
            necessary.
    
        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitely calculated) with the input
            `vector`.
    
        """
        # Extract ramp parameters if necessary (vector will be shortened!):
        vector = self.ramp.extract_ramp_params(vector)
        # Reset magdata and fill with vector:
        self.magdata.field[...] = 0
        self.magdata.set_vector(vector, self.data_set.mask)
        # Simulate all phase maps and create result vector:
        result = np.zeros(self.m)
        hp = self.hook_points
        mag_vec = self.magdata.field_vec
        res=[]
        phases=self.fast_jac(mag_vec)
        # add ramp params
        for i in range(self.data_set.count):
            res.append(np.ravel(phases[i,:,:])+self.ramp.jac_dot(i))
        res=np.concatenate(res)
        result[hp[0]:hp[self.data_set.count]]=res
        return result

    def fast_jac_T(self, phase_vector):
        kernel=self.phasemappers[0].kernel
        projector=self.projector
        n_proj=self.data_set.count
        mag_field=self.magdata
        
        padded_dim=(n_proj,)+kernel.dim_pad
        raw_dim=(n_proj,)+kernel.dim_uv
        padded_mag_slice=(slice(None),)+kernel.slice_mag
        padded_phase_slice=(slice(None),)+kernel.slice_phase

        phase_vector=cp.array(phase_vector)
        phase_adj = cp.zeros(padded_dim, dtype=kernel.u.dtype)
        phase_adj[padded_phase_slice] = phase_vector.reshape(raw_dim)
        phase_adj_fft = cpfft.rfft2(phase_adj)
        u_mag_adj_fft = phase_adj_fft * cp.conj(cp.array(kernel.u_fft)) 
        v_mag_adj_fft = phase_adj_fft * cp.conj(cp.array(kernel.v_fft))
        u_mag_adj = cpfft.irfft2(u_mag_adj_fft)[padded_mag_slice]
        v_mag_adj = cpfft.irfft2(v_mag_adj_fft)[padded_mag_slice]
        
        
        # transpose the projector
        geom_vectors=projector.pg['Vectors']
        det_vec_y=(geom_vectors[:,9:12]) #y
        det_vec_x=(geom_vectors[:,6:9]) #x
        dim_uv=(projector.pg['DetectorRowCount'],projector.pg['DetectorColCount'])
        
        axes=((0,0,1),(0,1,0),(1,0,0)) # (z ,y ,x)
        M_u=cp.asnumpy(cp.transpose(u_mag_adj, axes=[1,0,2]))
        M_v=cp.asnumpy(cp.transpose(v_mag_adj, axes=[1,0,2]))
        
        
        # # for inspection
        # M_u[:,:proj_i,:]=0
        # M_v[:,:proj_i,:]=0
        # M_u[:,proj_i+1:,:]=0
        # M_v[:,proj_i+1:,:]=0
        # plt.matshow(cp.asnumpy(M_u[:,proj_i,:]))
        # plt.matshow(cp.asnumpy(M_v[:,proj_i,:]))
        
        # to match pyramid conventions
        magdata=pr.VectorData(mag_field.a, np.zeros(mag_field.field.shape))
        mu_comp = projector.BP(M_u)[:,::-1,:]
        mv_comp = projector.BP(M_v)[:,::-1,:]
        
        # #more inspection
        # r.plot_reconstruction(mv_comp)
        # revf=pr.VectorData(mag_field.a, np.zeros(mag_field.field.shape))
        # revf.field[0,...]=mv_comp
        # revf.plot_quiver3d()
        
        for i in range(len(axes)):
            axis=axes[i] 
            
            coef_u = cp.dot(det_vec_y,axis) 
            coef_v = cp.dot(det_vec_x,axis)
        
            # pyramid projection weirdness
            coef_u,coef_v=coef_v,coef_u
            if i==1:
                coef_u=-1*coef_u
                coef_v=-1*coef_v
                
        
            #find contribution to one component in projection
            M_component_projection= M_u*coef_u[np.newaxis,:,np.newaxis] + M_v*coef_v[np.newaxis,:,np.newaxis]
            #backproject to volume of one vector component
            M_component = projector.BP(M_component_projection)[:,::-1,:] 
        
            magdata.field[-i-1,:,:,:] = M_component  #need reverse axis order for pyramid vector fields   
        return(magdata.field_vec)


    def jac_T_dot(self, x, vector):
        """'Calculate the product of the transposed Jacobi matrix with a given `vector`.
        Sped up version using CUDA graphics card acceleration.
        Uses self.fast_jac_T

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Evaluation point of the jacobi-matrix. The jacobi matrix is constant for a linear
            problem, thus `x` can be set to None (it is not used int the computation). Is used
            for the case that in the future nonlinear problems have to be solved.
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of all 2D phase maps one after another in one vector.

        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Product of the transposed Jacobi matrix (which is not explicitely calculated) with
            the input `vector`. If necessary, transposed ramp parameters are concatenated.

        """
        hp = self.hook_points
        phase_vec = vector[hp[0]:hp[self.data_set.count]]
        proj_T_result = self.fast_jac_T(phase_vec)
        self.magdata.field_vec = proj_T_result
        result = self.magdata.get_vector(self.data_set.mask)
        ramp_params = self.ramp.jac_T_dot(vector)  # calculate ramp_params separately!
        return np.concatenate((result, ramp_params))
    

    def __call__(self, x):
        return(self.jac_dot(None,x))

    def vfield_to_vector (self, vfield):
        """
        Covert 3D vector field to a 1D vector form used by minimisation algorithms
        Adds stored ramp parameters onto the end of the vector.
        
        returns: vector, np.1Darray
        """
        mask=self.data_set.mask
        vfield=vfield.field
        n=self.n
        
        x=vfield[0][mask]
        y=vfield[1][mask]
        z=vfield[2][mask]
        vec=np.concatenate((x,y,z))
        if self.ramp is not None:
            vec = np.concatenate((vec, np.ravel(self.ramp.param_cache))) #current best estimate of ramp
            # vec = np.concatenate((vec, self.ramp.jac_T_dot(self.data_set.phase_vec))) #not the same as the already optimised ramp.
         
        return(vec)

    def vector_to_vfield(self, vector):
        """
        Convert 1D vector used by minimisation algorithms to 3D vector field.
        Ignores extra phase ramp parameters at the end of the vector.
        
        returns: magnetisation, pr.VectorData
        """
        x=vector
        mask=self.data_set.mask
        arr_len=len(mask[mask])
        vec_data=pr.VectorData(1, np.zeros((3,) + mask.shape))
        vec_data.field[0,...][mask]=x[:arr_len]
        vec_data.field[1,...][mask]=x[arr_len:arr_len*2]
        vec_data.field[2,...][mask]=x[arr_len*2:arr_len*3]
        return(vec_data)

    def vector_to_phase(self, vector):
        """
        Convert 1D vector of phase values to an array of 2D phase images

        Returns: phase_array, np.3Darray
        """
        phase_array=np.reshape(vector, (self.data_set.count,)+self.data_set.phasemaps[0].dim_uv) #assuming all images are the same size
        return(phase_array)

#simulation
def make_datasetCUDA_from_vfield(vfield, projection_x_ang, projection_z_ang, camera_rotation=0, 
                             centre_shift=(0,0,0), dim_uv=None, verbose=False, plot_results=False):
    """
    Takes 3D magnetisation vector field and returns a simulated dataset of phase images and projectors.
    Creates astra.OpTomo projector that is stored in GPU memory.
        
    vfield: pyramid.VectorField
        Magneisation
    projection_x_ang: list
        sample tilt in degrees
    projection_z_ang: list
        sample z rotation in degrees
    camera_rotatoin: list (or int if constant), default: 0
        camera rotation in degrees
    centre_shift: tuple, default:(0,0,0)
        3D mask center shift relative to image centre
    dim_uv: tuple, default:None
        size of projected images. Defaults to 2D side length equal to 3D volume diagonal length.

    returns pr.DataSet containing phase maps and projectors.
    """

    #convert camera rotation to an array
    n_proj=len(projection_x_ang)
    try:
        camera_rotation[0]
    except:
        camera_rotation=[camera_rotation]*n_proj
        
    #create projector
    r=AstraReconstructor(None, vfield.get_mask(), projection_z_ang, projection_x_ang, camera_rotation, dim_uv=dim_uv, verbose=verbose)
    t=r.move_reconstruction_centre(pos=centre_shift)
    proj_geom=r.proj_geom
    vol_geom=r.vol_geom
    proj_id=astra.create_projector('cuda3d', proj_geom, vol_geom)
    projector = astra.OpTomo(proj_id)

    if dim_uv is None:
        dim_uv=(proj_geom['DetectorRowCount'],proj_geom['DetectorColCount'])

    #initialise dataset
    data_set=DataSetCUDA(vfield.a, vfield.dim, mask=vfield.get_mask(), projector=projector)

    masks=projector.FP(vfield.get_mask())
    masks=np.transpose(masks, axes=[1,0,2]) # transpose such that first axis is tilt angle
    masks=(masks>=0.5) #pixel is accepted if it is at least half-filled
    confidences=np.zeros((n_proj,)+dim_uv)
    confidences[:,1:-1,1:-1]=1 #edge most pixels have confidence=0
    temp_phases=np.zeros((n_proj,)+dim_uv)

    phasemaps=[]
    proj_info=[]
    for i in range(n_proj):
        prj=DummyProjector(dim=vfield.dim, dim_uv=dim_uv, tilt=np.radians(projection_x_ang[i]), 
                           rotation=np.radians(projection_z_ang[i]), camera_rotation=np.radians(camera_rotation[i]))
        proj_info.append(prj)
        pm=pr.PhaseMap(vfield.a, temp_phases[i,:,:], mask=masks[i,:,:], confidence=confidences[i,:,:])
        phasemaps.append(pm)
    data_set.append(phasemaps, proj_info)
    
    #initialise forward model
    #consider rewriting the forward model to operate primarily in non-vector form, and only convert in the cost function?
    fmodel = ForwardModelCUDA(data_set, projector, ramp_order=0)
    phase_vec=fmodel(fmodel.vfield_to_vector(vfield)) 
    phases=fmodel.vector_to_phase(phase_vec)

    for i in range(n_proj):
        data_set.phasemaps[i].phase=phases[i]
    
    if plot_results:
        data_set.plot_phasemaps()
    
    return(data_set)

#reconstruction/Tomography
def reconstruct_from_phasemaps_CUDA(data, projector,lam=1e-3, max_iter=100, ramp_order=1, 
                               verbose=True, plot_input=True, plot_results=True, b_0 = 1, 
                                   regulariser_type='exchange', mean=None, abs_tol=1e-20, rel_tol=1e-20,
                                   mag_0=None, reg_mask=None):
    """
    TODO: define all regularisers as forward models and use a sum of models to define total model.
        consider rewriting the forward model to operate primarily in non-vector form, and only convert in the cost function?
        
    Takes a dataset of phasemaps created by 'make_phasemap_dataset'functin and returns a 3D Magnetisation vector field
    Reconstructs the magnetisation.

    
    
    
    lam=1e-3 #regularisation parameter determining the weighting between measurements and regularisation. 
    max_iter=100
    ramp_order=1 #None or int. what order of phase ramp in 3D is present in the dataset.
    verbose=True
    plot_input=True
    plot_results=True
    kwargs passed on as phasemap.plot_phase(**kwargs)
    b_0=1 units of mag field in T.
    """

    fwd_model = ForwardModelCUDA(data, projector, ramp_order=ramp_order) #define a forward model. How are ramps implemented?
    if regulariser_type == 'amplitude':
        lam1,lam2 = lam
        reg1 = pre.AmplitudeRegulariser(data_mask=data.mask, reg_mask=reg_mask, lam=lam1, add_params=fwd_model.ramp.n)
        reg2 = pre.ExchangeRegulariser(data_mask=data.mask, lam=lam2, add_params=fwd_model.ramp.n)
        reg = pr.ComboRegularisator([reg1, reg2])
        if verbose:
            print("Regularising amplitude and exchange energy")
    elif regulariser_type == 'mean only':
        reg = AmplitudeMeanRegulariser(mean=mean, data_mask=data.mask, reg_mask=reg_mask, lam=lam, add_params=fwd_model.ramp.n)
        if verbose:
            print("Regularising set mean")
    elif regulariser_type == 'amplitude only':
        reg = pre.AmplitudeRegulariser(data_mask=data.mask, reg_mask=reg_mask, lam=lam, add_params=fwd_model.ramp.n)
        if verbose:
            print("Regularising amplitude only")
    elif regulariser_type == 'mean':
        lam1,lam2 = lam
        reg1 = AmplitudeMeanRegulariser(mean=mean, data_mask=data.mask, reg_mask=reg_mask, lam=lam1, add_params=fwd_model.ramp.n)
        reg2 = pre.ExchangeRegulariser(data_mask=data.mask, lam=lam2, add_params=fwd_model.ramp.n)
        reg=pr.ComboRegularisator([reg1, reg2])
        if verbose:
            print("Regularising set mean amplitude and exchange energy")
    elif regulariser_type == 'exchange':
        reg = pre.ExchangeRegulariser(data_mask=data.mask, lam=lam, add_params=fwd_model.ramp.n)
        if verbose:
            print("Regularising exchange energy")
    else:
        #maybe not needed?
        reg = pr.FirstOrderRegularisator(data.mask, lam=lam, add_params=fwd_model.ramp.n) #define the regularisator order 
        if verbose:
            print("Regularising nearest neighbour difference squared")
    cost = pr.Costfunction(fwd_model, reg, track_cost_iterations=1) #define the cost function

    if mag_0 is None:
        mag_0=pr.VectorData(data.a, np.zeros((3,)+data.mask.shape))
        
    x=cost.fwd_model.vfield_to_vector(mag_0)
    current_cost=cost(x)
    print(f'Cost before optimization: {current_cost:7.5e}, model: {cost.chisq_m[-1]:7.5e}, regulariser: {cost.chisq_a[-1]:7.5e} ')
    
    # Reconstruct and save:
    magdata_rec = pr.reconstruction.optimize_linear(cost, max_iter=max_iter, verbose=verbose, 
                                                    abs_tol=abs_tol, rel_tol=rel_tol, mag_0=mag_0)
    # Finalize ForwardModel (returns workers if multicore):
    fwd_model.finalize()

    x=cost.fwd_model.vfield_to_vector(mag_0)
    current_cost=cost(x)
    print(f'Cost after optimization:  {current_cost:7.5e}, model: {cost.chisq_m[-1]:7.5e}, regulariser: {cost.chisq_a[-1]:7.5e} ')
    
    # Plot results:
    if plot_results:
        mag_ang_plot=pr.VectorData(magdata_rec.a, magdata_rec.field)
        mag_ang_plot = np.sign(mag_ang_plot.field)  * np.sqrt(np.abs(mag_ang_plot))
        mag_ang_plot.plot_quiver3d('Reconstructed Distribution (angle)')
        magdata_rec.plot_quiver3d('Reconstructed Distribution (amplitude)', coloring='amplitude')
        pu.matshow_n([magdata_rec.field_amp[data.dim[0]//2,:,:], np.sum(magdata_rec.field[0,:,:,:],axis=0)], 
                  ["magnitude in z slice", "M_x sum along z"], origin='lower')
                  
        
    return(magdata_rec, cost)

