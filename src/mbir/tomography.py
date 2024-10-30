# -*- coding: utf-8 -*-
# Aurys Silinga, 2024
#

"""
Tomographic tilt series alignment for mbir-tem
"""
# plotting and reading/writing files
import matplotlib.pyplot as plt
import numpy as np
import pyramid as pr
import astra
import copy

# image processing
import skimage.transform as sktr

# fitting algorithms
import scipy.optimize as op

from scipy.spatial.transform import Rotation as R
from .util import subplots_n, sktr_translate
from .alignment import find_image_shifts_correlation, find_image_shifts_com

class AstraReconstructor(object):
    """
    Stores the reconstruction parameters and collects garbage after reconstructions 
    Recon data defines the starting distribution for iterative CT algorithms
    """

    def __init__(self, proj_data=None, recon_data=None, projection_z_ang=0, projection_x_ang=0, camera_rotation=0, pixel_size=1,
                 recon_iters=50, algorithm='SIRT3D_CUDA', dim_uv=None, verbose=True):
        
        #Define reconstruction dimensions
        if (proj_data is None) and (recon_data is None):
            raise ValueError("Either projection or 3D model dataset needs to be provided to define reconstruction dimensions")
        
        if proj_data is not None:  
            proj_data=proj_data.copy()
            dim_yd, n_proj, dim_xd = proj_data.shape 
            dim_uv=(dim_yd,dim_xd)
            if recon_data is None:
                #dimensions of largest reconstructable object
                hypot=np.hypot(dim_yd,dim_xd)
                largest_total_rotation=np.max(np.abs( np.array(projection_z_ang)+np.array(camera_rotation) ))
                max_dim=round(hypot*np.abs(np.cos(np.radians(45-largest_total_rotation%90)))) 
                recon_data=np.zeros((max_dim,max_dim,max_dim))
            else:
                recon_data=recon_data.copy()
        elif recon_data is not None: #and proj data is missing
            recon_data=recon_data.copy()
            if dim_uv is None:
                max_dim=int(np.sqrt(2)*np.max(recon_data.shape))+1 #closest integer number legth of the diagonal
                dim_uv=(max_dim,max_dim) #detector dimensions (yd,xd)

        self.dim_uv=dim_uv
        self.proj_data=proj_data #can be none
        self.recon_data=recon_data
        self.projection_z_ang=projection_z_ang
        self.projection_x_ang=projection_x_ang
        self.camera_rotation=camera_rotation
        self.a_pix = pixel_size
        self.algorithm=algorithm
        self.recon_iters=recon_iters
        self.verbose=verbose
        
        # astra reconstruction volume definition
        dimz,dimy,dimx=self.recon_data.shape
        self.vol_geom = astra.create_vol_geom(dimy,dimx,dimz) #yxz
        # astra beam path definition for defining projection matrices
        self.proj_geom = self.create_astra_projection_geometry(self.dim_uv, self.projection_z_ang, self.projection_x_ang, 
                                                               self.camera_rotation, self.a_pix)

        if proj_data is not None:
            n_proj=self.proj_geom['Vectors'].shape[0]
            assert n_proj==proj_data.shape[1], "The number of projection angles does not match the number in the dataset!"

    def reconstruct(self, algorithm=None, recon_iters=None, plot_results=False):
        """
        do the CT reconstruction and perform garbage collection in the GPU.

        algorithm : str, optional
            One of the following string values. Defaults to algorithm chosen on reconstructor creation.
    
            'CGLS3D_CUDA'
    
            'SIRT3D_CUDA'
    
            'BP3D_CUDA'
            
        """
        if algorithm is not None:
            self.algorithm=algorithm
        if recon_iters is not None:
            self.recon_iters=recon_iters
        
        # astra dataset definitions with projection models
        proj_data_id = astra.data3d.create('-sino', self.proj_geom, self.proj_data)
        rec_id = astra.data3d.create('-vol', self.vol_geom, self.recon_data)
        
        # Set up the parameters for a reconstruction algorithm using the GPU
        cfg = astra.astra_dict(self.algorithm)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_data_id
        
        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)

        # Run #recon_iters iterations of the algorithm
        if self.verbose:
            print(f"Running {self.algorithm} reconstruction for {self.recon_iters} iterations.")
        astra.algorithm.run(alg_id, self.recon_iters)
        
        # Get the result
        rec = astra.data3d.get(rec_id)
        
        # Clean up. Note that GPU memory is tied up in the algorithm object,
        # and main RAM in the data objects.
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_data_id)

        if plot_results:
            self.plot_reconstruction(rec)
        
        return(rec)

    def project(self, recon_data=None, plot_results=False):
        if recon_data is None:
            recon_data = self.recon_data
        proj_id, proj_data = astra.create_sino3d_gpu(recon_data, self.proj_geom, self.vol_geom)
        astra.data3d.delete(proj_id)

        if plot_results:
            self.plot_projections(proj_data)
        return (proj_data)
        
    def create_astra_projection_geometry(self, dim_uv, projection_z_ang=0, projection_x_ang=0, camera_rotation=0, a_pix=1):
        """
        Generate the same geometry as in pyramid, but by using vector specification in Astra
        """ 
        #if some rotations are constant, reshape them into arrays
        try:
            n_proj = len(projection_x_ang)
        except:
            projection_x_ang = [projection_x_ang]
            n_proj = len(projection_x_ang)
        if isinstance(projection_z_ang, (float, int)):
            projection_z_ang=[projection_z_ang]*n_proj
        if isinstance(camera_rotation, (float, int)):
            camera_rotation=[camera_rotation]*n_proj
        
        angles = tuple(zip(projection_z_ang, projection_x_ang))
        
        #starting, default Astra, orientation #xyz order
        b=[0,-1,0] # beam direction
        xd=[1,0,0] # detector x
        yd=[0,0,1] # detector y
        cd=[0,0,0] # detector center relative to beam
        
        r1=R.from_euler('y', (camera_rotation), degrees=True) # detector rotation
        r2=R.from_euler('x', (90), degrees=True)  # sarting orientation as in pyramid (beam along -z direction in the 3Dmodel)
        r3_array=R.from_euler('ZX', angles, degrees=True) #sample z-rotation and subsequent x-tilt 
        
        r_beam=r3_array*r2
        r_detector = r_beam*r1
        
        vectors = np.zeros((len(angles), 12))
        
        # beam directions
        vectors[:,0:3]=r_beam.apply(b)
        # center of detector
        vectors[:,3:6] = cd
        # vector from detector pixel (0,0) to (0,1) i.e. detector x-axis unit vector
        vectors[:,6:9] = r_detector.apply(xd) *a_pix
        # vector from detector pixel (0,0) to (1,0) i.e. detector y-axis unit vector
        vectors[:,9:12] = r_detector.apply(yd) *a_pix
        
        # Parameters: #rows (yd), #columns (xd), vectors
        proj_geom = astra.create_proj_geom('parallel3d_vec', dim_uv[0], dim_uv[1], vectors)
        return (proj_geom)

    def geom_jitter(self, shifts, proj_geom=None):
        """Apply a postalignment to a vector-based projection geometry.
        Can be used to model the rotation axis offset and image jitter.
    
        For 2D geometries, the argument factor is a single float specifying the
        distance to shift the detector (measured in detector pixels).
    
        For 3D geometries, factor can be a pair of floats specifying the horizontal
        resp. vertical distances to shift the detector. If only a single float
        is specified, this is treated as a horizontal shift.
    
        :param proj_geom: input projection geometry
        :type proj_geom: :class:`dict`
        :param shifts: number of pixels to shift the detector in dx and dy respectively, for each projection
            [(dy1,dx1),...,(dyN,dxN)]

        proj_geom: dict, default: None
            configuration dictionary defning the projection geometry.
            if None, uses and changes self.proj_geom
        """
        if proj_geom is None:
            proj_geom=self.proj_geom
            if self.verbose:
                print("Applying detector shift to self")
        else:
            proj_geom=copy.deepcopy(proj_geom)
    
        if proj_geom['type'] == 'parallel3d' or proj_geom['type'] == 'cone' or proj_geom['type'] == 'parallel' or proj_geom['type'] == 'fanflat':
            proj_geom = astra.geom_2vec(proj_geom)
        
        try:
            dy,dx=shifts[0]
            is_shifts=True
        except:
            is_shifts=False
            
        if is_shifts: #translate each detector differently
            if proj_geom['type'] == 'parallel3d_vec' or proj_geom['type'] == 'cone_vec':
                V = proj_geom['Vectors']
                for i in range(V.shape[0]):
                    dy,dx = shifts[i]
                    V[i,3:6] = V[i,3:6] + dx * V[i,6:9]
                    V[i,3:6] = V[i,3:6] + dy * V[i,9:12]
    
        else: #translate all the same
            dy,dx=shifts
            if proj_geom['type'] == 'parallel3d_vec' or proj_geom['type'] == 'cone_vec':
                V = proj_geom['Vectors']
                V[:,3:6] = V[:,3:6] + dx * V[:,6:9]
                if len(factor) > 1:  # Accommodate factor = (number,) semantics
                    V[:,3:6] = V[:,3:6] + dy * V[:,9:12]
            else:
                raise RuntimeError(proj_geom['type'] + 'geometry is not suitable for postalignment')
                
        return proj_geom

    def move_reconstruction_centre(self, pos, is_relative=True, vol_geom=None):
        """
        changes the centre of the astra volume geometry to a new position defined by 'pos'. 
        Can be used to shift the point of rotation away from the centre of the model.
        
        pos: tuple of float
            new position in (x,y,z)
        is_relative: bool, default: True
            Makes pos define a centre shift instead of absolute value. 
            Has opposite sign from shift definitions in pyramid
        
        """
        if vol_geom is None:
            vol_geom0=self.vol_geom
        vol_geom1=astra.move_vol_geom(vol_geom0, pos, is_relative=is_relative) #implement being off-centre

        if vol_geom is None:
            self.vol_geom=vol_geom1
            if self.verbose:
                print("Applying centre shift to self")
        return(vol_geom1)

    def cleanup(self, verbose=None):
        """
        Removes all astra objects from RAM and GPU memories.
        If verbose==True, then lists all object in memory before deleting.
        """
        if verbose is None:
            verbose=self.verbose
        if verbose:
            print("All Astra objects in memory")
            astra.matrix.info()
            astra.data3d.info()
            astra.data2d.info()
            astra.algorithm.info()
            astra.projector.info()
    
            print("Deleting all objects")
        astra.clear()

        if verbose:
            print("All Astra objects in memory after deletion")
            astra.matrix.info()
            astra.data3d.info()
            astra.data2d.info()
            astra.algorithm.info()
            astra.projector.info()

    def plot_reconstruction(self, reconstruction=None, vmin=None, vmax=None, title=""):
        if reconstruction is None:
            rec = self.recon_data
            if self.verbose:
                print("fetching 3D data from self")
        else:
            rec=reconstruction
        axis_string='zyx'
        labels=[]
        imgs=[]
        labels=["z sum projection \n x", "y sum projection \n x", "x sum projection \n y"]
        for axis in range(len(rec.shape)):
            imgs.append(np.sum(rec, axis=axis))
            #labels.append(axis_string[axis]+" sum projection")
        subplots_n(imgs, labels, title, vmin=vmin, vmax=vmax)

    def plot_projections(self, projections=None, angles=None, vmin=None, vmax=None, title=""):
        if projections is None:
            p=self.proj_data
            if self.verbose:
                print("fetching projection data from self")
        else:
            p=projections
            
        if angles is None:
            angles=self.projection_x_ang
            if self.verbose:
                print("fetching angles from self")
            
        for i in range(p.shape[1]):
            plt.matshow(p[:,i,:], vmin=vmin, vmax=vmax)
            plt.title(title+f" projection at {angles[i]:.2f} deg tilt")

    def mask_edge(self, projections=None, ramp_width=3):
        """
        Set the edge-most pixels of projection images to 0 and smooth the transition with a linear ramp.
        ramp_width=3 gives [0, 0.5, 1, 1, ..., 1, 1, 0.5, 0] intensity multipliers
        """
        if projections is None:
            p=self.proj_data
            if self.verbose:
                print("fetching projection data from self")
        else:
            p=projections

        p=np.array(p).copy()   
        
        #make mask
        img0=p[:,0,:]
        mask=np.zeros(img0.shape)
        vals=np.linspace(0,1,num=ramp_width)
        for j in range(ramp_width):
            if j==0:
                mask[:,:]=vals[j]
                continue
            mask[j:-j,j:-j]=vals[j]

        # apply edge mask
        for i in range(p.shape[1]):
            img=p[:,i,:]
            p[:,i,:] = img*mask

        if projections is None:
            self.proj_data=p
            if self.verbose:
                print("Applying edge mask to self")
        return p

    def mask_reconstruction_circle(self, reconstruction=None, radius_extension=-1, plot_results=False):
        """
        Set to 0 everything that is outside the central circle in the reconstruction
        """
        if reconstruction is None:
            rec = self.recon_data
            if self.verbose:
                print("fetching 3D data from self")
        else:
            rec=reconstruction.copy()
        
        slice=rec[:,:,0]
        dy,dx=slice.shape
        yv,xv = np.mgrid[:dy,:dx]
        yv=yv-(dy/2-0.5)
        xv=xv-(dx/2-0.5)
        dist=np.hypot(yv,xv)
        valid_values=dist<(np.min((dy,dx))/2+radius_extension)
    
        if plot_results:
            plt.matshow(valid_values)
            plt.title("circular mask on x-axis")
    
        rec[np.bitwise_not(valid_values),:]=0 #all invalid values are set to 0
    
        if reconstruction is None:
            self.recon_data=rec
            if self.verbose:
                print("Applying circular mask to self")
        return(rec)

    def rotate_detector(self, angle, proj_geom=None, degrees=True):
        """
        Detector rotation around the beam axis by 'angle' in degrees.
        projection_geometry==None applies rotation to self.proj_geom.
        returns: projection_geometry with rotated detector specification
        
        """
        if proj_geom is None:
            proj_geom=self.proj_geom
            if self.verbose:
                print("Applying detector rotation to self")
        else:
            proj_geom=copy.deepcopy(proj_geom)
    
        if proj_geom['type'] == 'parallel3d':
            proj_geom = astra.geom_2vec(proj_geom)
        
        V=proj_geom['Vectors']
        for i in range(V.shape[0]):
            beam_direction=V[i,0:3]
            rotation_vector = -1*beam_direction/np.linalg.norm(beam_direction) #default beam direction is -z
            r = R.from_rotvec(angle*rotation_vector, degrees=degrees)
            detector_xaxis = V[i,6:9]
            detector_yaxis = V[i,9:12]
            V[i,6:9]=r.apply(detector_xaxis)
            V[i,9:12]=r.apply(detector_yaxis)


class CostFunction(object):
    """
    Class for storing a cost function and supporting information

    get_cost(x): 
        returns the cost for x
    params_to_vec(params):
        returns x for specific params
    vec_to_params(x):
        returns params from  specifitc x
    
    bounds: tuple
        bounds used for scipy function minimisation
    x0: tuple
        starting parameter vector for the cost function  
    
    """
    def __init__(self, get_cost=None,  bounds=None, x0=None, params_to_vec=None, vec_to_params=None):
        self.get_cost=get_cost
        self.bounds=bounds
        self.x0=x0
        self.params_to_vec=params_to_vec
        self.vec_to_params=vec_to_params


class ContrastAligner(object):
    """
    Keeps track of geometries and metadata during alignment optimisation

    Stores the best estimate of alignments (camera rotation and image shift).
    Every call of an alignment method starts at previously set best estimate.

    Meant for maximising the contrast of a subvolume of the reconstruction.
    Only works for a single tilt-axis.
    Works best on a subvolume containing sharp edges.

    Projection definition. Rotate model around beam axis by 'projection_z_ang' deg,
    then tilt model around x-axis by 'projection_tilt' deg, 
    then rotate image around beam axis by 'camera_rotation' deg.
    """
    def __init__(self, proj_shifted_noised, projection_x_ang, projection_z_ang=0, camera_rotation=0, vol=None, recon_iters=50, 
                       centre_point=None, centre_width=None, shifts=None, centre_shift=(0,0,0), algorithm='SIRT3D_CUDA', verbose=False):
        self.history=[] # [(a,(sy,sx)),...]
        self.proj_shifted_noised=copy.deepcopy(proj_shifted_noised)
        self.vol=copy.deepcopy(vol)
        self.projection_z_ang=copy.copy(projection_z_ang)
        self.projection_x_ang=copy.copy(projection_x_ang)
        self.camera_rotation=camera_rotation
        self.recon_iters=recon_iters
        self.centre_point=centre_point
        self.centre_width=centre_width
        if shifts is None:
            self.shifts_current = [(0,0)]*len(self.projection_x_ang)
        else:
            self.shifts_current=copy.deepcopy(shifts)
        self.centre_shift=centre_shift[::-1] # corrected to (dz,dy,dx)
        self.history.append(('starting alignment', self.camera_rotation, self.shifts_current))
        self.algorithm='SIRT3D_CUDA'
        self.slice_averaging_width=1
        self.slice_indices=None
        self.verbose=verbose
    
    def cross_correlation_consecutive(self, upsample_factor=1, sort=True):
        """
        Starting point for alignment, capable of subpixel precision. 
        Only works if images are similar enough, which needs <5 deg tilt spacings
        """
        
        imgs=np.transpose(self.proj_shifted_noised, axes=[1,0,2])

        if sort: # make projections in order of tilt angle
            si=np.argsort(self.projection_x_ang) # sorting index
            usi=np.argsort(si) # unsorting index
            
            s_imgs=imgs[si] #sorted array of images
            
            s_shifts=find_image_shifts_correlation(s_imgs, upsample_factor=upsample_factor)
            shift_estimates=s_shifts[usi]
        else:
            shift_estimates=find_image_shifts_correlation(imgs, upsample_factor=upsample_factor)


        
        # save the results
        shift_estimates=np.array(shift_estimates)

        if self.verbose:
            print("Consecutive cross correlation detected shifts")
        camera_rotation, shifts=self.get_current_alignment(shifts=shift_estimates, print_values=self.verbose) #for display

        return(shifts)

        
    def centre_of_mass(self, subpixel=True, test_image_index=None):
        """
        Measures the position of the CoM in each image, relative to CoM position in the selected 'test' image.

        Returns: (camera_rotation_estimate, image_drift_estimate)
        """
        imgs=np.transpose(self.proj_shifted_noised, axes=[1,0,2])

        shift_estimates=find_image_shifts_com(imgs, test_image_index=test_image_index, subpixel=subpixel)

        # save the results
        shift_estimates=np.array(shift_estimates)

        if self.verbose:
            print("Centre of mass detected shifts")
        camera_rotation, shifts=self.get_current_alignment(shifts=shift_estimates, print_values=self.verbose) #for display

        return(shifts)

        
    def get_reconstructor(self, camera_rotation=None, shifts=None, verbose_reconstructor=False):
        """
        Returns a reconstructor object to be used for alignment

        recomended to turn the recontructor verbose print statements off when using it for optimisation
        """
        if camera_rotation is None:
            camera_rotation = self.camera_rotation
        if shifts is None:
            shifts=self.shifts_current
        
        reconstructor= AstraReconstructor(self.proj_shifted_noised, self.vol, self.projection_z_ang, self.projection_x_ang, camera_rotation, 
                                      recon_iters=self.recon_iters, verbose=verbose_reconstructor, algorithm=self.algorithm)
        reconstructor.mask_edge()
        reconstructor.move_reconstruction_centre(self.centre_shift)
        reconstructor.geom_jitter(shifts)
        
        return(reconstructor)


    def get_volume_slice(self, volume=None):
        """
        Returns 3D slice object for selecting a region of the 3D reconstruction
        
        if volume is given and centre point is not specified, selects the centre point automatically.

        Returns: (slice())*3
            Slice object for 3D volume
        """
        # load params
        centre_point=self.centre_point
        centre_width=self.centre_width

        if (centre_point is None) and (volume is not None):
            dz,dy,dx=np.array(volume).shape
            centre_point=(dz//2,dy//2,dx//2)
        
        # define the slices
        if (centre_point is None) or (centre_width is None):
            volume_slice = tuple([slice(None) for i in range(3)])
            if self.verbose:
                print("Subvolume is not specified. Returning full volume instead.")
        else:
            volume_slice=tuple([slice(np.max((centre_point[i]-centre_width[i]//2, 0)),centre_point[i]+centre_width[i]//2) for i in range(3)])

        return(volume_slice)


    def get_subvolume(self, plot_result=False, vmin=-1, vmax=None,  plot_full_volume=False, title=""):
        """
        Perform the reconstruction using currently stored best estimates and slice out the defined sub-volume.
        If plotting is set to True, can also plot the subvolume and the full reconstruction.

        plot_result: bool, default: False
            If subvolume should be plotted.
        plot_full_volume: bool, default: False
            If full reconstruction volume should be plotted
        vmin: int, default: -1
            Plot colourmap minimum value.
        vmax: int, default: None
            Plot colourmap maximum value.
        title: string, default ""
            Title prefix for subvolume plot.

        Returns: 3Darray
            Subvolume
        """
        #current best reconstruction estimate
        reconstructor=self.get_reconstructor()
        rec=reconstructor.reconstruct(plot_results=False)

        #slice the reconstruction to get the subvolume
        volume_slice=self.get_volume_slice(volume=reconstructor.recon_data)
        sub_vol = rec[volume_slice]
        
        #plotting
        if plot_full_volume:
            reconstructor.plot_reconstruction(rec, title=title+"Full Reconstruction", vmin=vmin, vmax=vmax)
        if plot_result:
            reconstructor.plot_reconstruction(sub_vol, title=title+"Sub Volume", vmin=vmin, vmax=vmax)

        return(sub_vol)
    
    def get_projections(self, plot_result=False, projection_slice=slice(None), title=""):
        """
        Correct all projections to reflect currently stored alignment and return them. 
        If plot_result = True, also plots the projections.

        plot_result: bool, default: False
            If projections should be plotted.
        title: string, default ""
            Title plot.
        projection_slice: slice(), default slice(None)
            Defined a subset of projections to plot.
            

        Returns: list of 2Darray
            Projection images
        """
        projections=self.trasform_images()
        r=AstraReconstructor(projections[:,projection_slice,:], None, 0, self.projection_x_ang[projection_slice], 0)
        r.plot_projections(title=title)
        return(projections[:,projection_slice,:])
    
    def trasform_images(self):
        """
        Applies the currently stored detector rotations and projection shifts to correct the projection images.
        Unrotates and unshifts every projection. 

        Returns: list of images
            Corrected projection images.
        """
        imgs=np.transpose(self.proj_shifted_noised, axes=[1,0,2])
        imgs_transformed=[]
        for i,im in enumerate(imgs):
            dy,dx=self.shifts_current[i]
            im=sktr_translate(im, dy=dy, dx=dx)
            im=sktr.rotate(im,self.camera_rotation)
            imgs_transformed.append(im)
        imgs_transformed=np.transpose(imgs_transformed, axes=[1,0,2])
        return(imgs_transformed)

    def get_current_alignment(self, camera_rotation=None, shifts=None, print_values=False):
        """
        Returns the currect estimate of camera rotation in degress and image drift in pixels.
        if print_values=True, prints the current alignment values.
        """
        if camera_rotation is None:
            camera_rotation=self.camera_rotation
        if shifts is None:
            shifts=self.shifts_current
        
        if print_values:
            print("detector rotation, (deg):",f"{camera_rotation:.2f}")
            print("shift y (pix):",[f"{s[0]:4.2f}" for s in shifts])
            print("shift x (pix):",[f"{s[1]:4.2f}" for s in shifts])

        return(camera_rotation, shifts)

    # NOT TESTED. Usually gives wrong result because global cost minimum is when images are outside FOV.
    # def minimise_shgo(self, cost_func, bounds, x0=None, local_iter=1, global_iter=1, local_method='Powell'):
    #     """
    #     NOTE: Does not actually respect bounds. TOO good at finding global minima. Could add multiprocessing with scipy>1.10.
        
    #     Calls scipy.optimise.shgo and inputs the iteration number parameters.
    #     returns the scipy..OptimisationResult

    #     Local methods: 'Nelder-Mead' or 'Powell'
    #     """
    #     local_options={
    #         'maxiter': local_iter,
    #         'bounds':bounds
    #     }
        
    #     minimizer_kwargs={
    #         'method':local_method, 
    #         'options':local_options
    #     }
        
    #     global_options={
    #         'maxiter': global_iter,
    #     }
    #     if x0 is None:
    #         x0=np.mean(bounds,axis=1)
    #     if self.verbose:
    #         print('Starting costfunction value:', cost_func(x0))
        
    #     res = op.shgo(cost_func, bounds, minimizer_kwargs=minimizer_kwargs, options=global_options)

    #     if self.verbose:
    #         print('Final costfunction value:', cost_func(res.x))
    #         print(res)

    #     return(res)

    # def minimise_hopping(self, cost_func, bounds, x0=None, local_iter=1, global_iter=1, local_method='Powell'):
    #     """
    #     NOTE: Does not actually respect bounds. TOO good at finding global minima.
        
    #     Calls scipy.optimise.shgo and inputs the iteration number parameters.
    #     returns the scipy..OptimisationResult

    #     Local methods: 'Nelder-Mead' or 'Powell'
    #     """
    #     local_options={
    #         'maxiter': local_iter,
    #         'bounds':bounds
    #     }
        
    #     minimizer_kwargs={
    #         'method':local_method, 
    #         'options':local_options
    #     }
        
    #     global_options={
    #         'maxiter': global_iter,
    #     }
    #     if x0 is None:
    #         x0=np.mean(bounds,axis=1)
    #     if self.verbose:
    #         print('Starting costfunction value:', cost_func(x0))
        
    #     res = op.basinhopping(func, x0, niter=global_iter, T=0.1, stepsize=1, minimizer_kwargs=minimizer_kwargs, interval=10)
                
    #     if self.verbose:
    #         print('Final costfunction value:', cost_func(res.x))
    #         print(res)

    #     return(res)

    # def minimise_annealing(self, cost_func, bounds, x0=None, local_iter=1, global_iter=5, local_method='Powell'):
    #     """        
    #     Calls scipy.optimise.dual_annealing and inputs the iteration number parameters.
    #     returns the scipy..OptimisationResult

    #     Local methods: 'Nelder-Mead' or 'Powell'
    #     """
    #     local_options={
    #         'maxiter': local_iter,
    #     }
        
    #     minimizer_kwargs={
    #         'method':local_method, 
    #         'options':local_options
    #     }
        
    #     if x0 is None:
    #         x0=np.mean(bounds,axis=1)
    #     if self.verbose:
    #         print('Starting costfunction value:', cost_func(x0))
        
    #     res =  op.dual_annealing(cost_func, bounds, minimizer_kwargs=minimizer_kwargs, x0=x0, maxiter=global_iter)

    #     if self.verbose:
    #         print('Final costfunction value:', cost_func(res.x))
    #         print(res)

    #     return(res)

    def minimise_local(self, cost_func, bounds=(-np.inf, np.inf), x0=None, local_iter=1, local_method='Powell'):
        """        
        Calls scipy.optimise.minimise using 'Powel' algorithm and inputs the minimisation parameters.
        returns the scipy..OptimisationResult

        Local methods: 'Nelder-Mead' or 'Powell'
        """
        local_options={
            'maxiter': local_iter,
        }
        
        if isinstance(cost_func, CostFunction):
            bounds=cost_func.bounds
            x0 = cost_func.x0
            cost_func = cost_func.get_cost
        
        
        if x0 is None:
            x0=np.mean(bounds,axis=1)
        if self.verbose:
            print('Starting costfunction value:', cost_func(x0))
        
        res =  op.minimize(cost_func, x0, bounds=bounds, options=local_options, method=local_method)

        if self.verbose:
            print('Final costfunction value:', cost_func(res.x))
            print(res)

        return(res)

    # NOT TESTED
    # # IF ALIGNING ONE LINE AT A TIME
    # def get_sinogram_reconstructor2D(self, sino, verbose_reconstructor=False):
    #     """
    #     reconstructor reshaped to accept sinograms instead of 3D volumes.
    #     """
    #     #make 3D
    #     s=np.zeros((*sino.shape,1))
    #     s[:,:,0]=sino
    #     sino=s
    #     vol_slice=np.zeros((sino.shape[0],sino.shape[0],1))
        
    #     reconstructor= AstraReconstructor(sino, vol_slice, 0, self.projection_x_ang, 0, recon_iters=self.recon_iters, 
    #                                       verbose=verbose_reconstructor, algorithm=self.algorithm)
    #     return(reconstructor)


    def get_sinogram(self, index=None, plot_result=False, title=""):
        """
        Returns a sinogram at specified x index. Plots it if plot_result = True.
        Sinogram is sliced from projection images corrected according to stored alignment.

        index: int, default: None
            x-coordinate of the sinogarm
        plot_results: bool, default: False
            if the sinogram should be plotted
        title: string, default: ""
            Title prefix for the plot.
        
        Returns: 2Darray
            The sinogram.
        """
        imgs=self.trasform_images()
        if index is None:
            if self.verbose:
                print("Index of sinogram x coordinate is not defined. Showing the middle corrdinate.")
            index=imgs.shape[2]//2 #use middle of image

        #sort the images in order of tilt angle
        imgs = np.transpose(imgs, axes=[1,0,2])
        si=np.argsort(self.projection_x_ang+self.projection_z_ang*1000) # sorting index, corrected for multiple tilt axes
        s_imgs=imgs[si] #sorted array of images
        imgs = np.transpose(s_imgs, axes=[1,0,2])
        s=imgs[:,:,index]

        if plot_result:
            #plotting axes are wrong if multiple tilt axes are used.
            plt.matshow(s, extent=[np.min(self.projection_x_ang), np.max(self.projection_x_ang), 0, s.shape[0]]) 
            plt.xlabel("tilt angle, deg")
            plt.ylabel("y, pix")
            plt.title(title+"Sinogram at x = "+str(index))
            
        return (s)

    def get_slices(self, indices=None, thickness=None, circle_mask_edge=-1, plot_result=True, title=""):
        """
        Return and plot reconstruction slices.
        
        indices: list
            which slices to plot. Axis alignment needs 3 slices, but can work with 2. 
        thickness: int, default None
            How many slices to average to reduce noise
        circle_mask_edge: int, default -1
            Masks the outside the circle where projections do not overlap. e.g. -4 masks outside the cirlcle 
            and 4 pixels from the edge inside the circle.
        plot_result: bool, default: False
            If the slices should be plotted
        title: string, default: ""
            Title prefix for the plot.
        
        Returns: list of 2D ndarray. 
            Reconstruction slices 
        
        """
        if indices is None:
            indices=self.slice_indices
        if thickness is None:
            thickness = self.slice_averaging_width
        
        r=self.get_reconstructor()
        rec=r.reconstruct()
        rec=r.mask_reconstruction_circle(rec, circle_mask_edge)
        
        if indices is None:
            if self.verbose:
                print("Relevant reconstruction slices are not defined. Showing first, middle, and last slice.")
            indices=(0+2, rec.shape[2]//2, rec.shape[2]-3-thickness) ## use edge slices that are not in the masked region
            self.slice_indices=indices
        
        rec_slices=[]
        for i in indices:
            rec_slice=np.average(rec[:,:,i:i+thickness],axis=2) #get a slice
            rec_slices.append(rec_slice)
        if plot_result:
            labels=["slice at x = "+str(i) for i in indices]
            subplots_n(rec_slices, labels=labels, title=title+"Reconstruction slices")
        
        return (rec_slices)
         

    def costfunction_variance(self, sub_vol):
        """better for volumes"""
        # create a constructor for given subvolumes
        ret=np.var(np.abs(sub_vol)) # physical variance
        return(float(-ret)) # float datatype works with more minimisers
        
    def costfunction_entropy(self, image, hist_min, hist_max, bins=64):
        """better for individual slices"""
        if (np.max(image) > hist_max) or (np.min(image)<hist_min):
            raise ValueError("Histogram range unsuitable for entropy calculation.")
        
        hist, bin_edges = np.histogram(image, bins=bins, range=[hist_min, hist_max])
        hist=hist/image.size + 1e-30 #why the size-based normalisation? it should be constant for all slices in the same function
        cost = -np.dot(hist, np.log2(hist))
        return(cost)


    
    def get_costfunction_rotation_centre(self, indices=None, circle_mask_edge=-1, thickness=None, n_bins=64, 
                                        plot_input=False):
        """
        From RL thesis on removing curving on multiple slices.
        
        Cost function for finding the axis of rotation. 
        Minimises entropy of multiple slices

        thickness: int
            how many slices to average to get represent one slice
        
        returns (cost_fun, bounds, x0, vectorise_params)

        TODO: add automatic slice chooser
        """
        if indices is None:
            indices=self.slice_indices
        if thickness is None:
            thickness = self.slice_averaging_width
        
        #for keeping the shifts consistent
        base_shifts=np.array(copy.deepcopy(self.shifts_current))
        base_ang=copy.copy(self.camera_rotation)
        
        r=self.get_reconstructor()
        test_rec=r.reconstruct()

        #bounds for histogram entropy estimation
        hist_max = np.max(test_rec) * 2
        hist_min = np.min(test_rec) * 2
        if hist_min > 0:
            hist_min=0

        #maxium shift is 1/5 image width (considering extension when rotated), or at least one pixel
        bound_side=np.max((1,np.max(test_rec.shape)//5)) 
        bounds=[(-90,90),(-bound_side, bound_side)] # starting bounds

        n_proj=len(self.projection_x_ang)
                
        def params_to_vec(camera_rotation, shifts):
            """
            convert rotation axis angle and perpedicular shift to a parameter vector.
            """
            ds=np.array(shifts)-np.array(base_shifts)
            #y axis is perpedicular to tilt axis, after rotation axis is corrected
            dsy=np.mean([s[0] for s in ds]) 
            return (camera_rotation,dsy)
            
        def vec_to_params(x):
            """
            convert parameter vector to rotation axis angle and perpedicular shift.
            """
            a=x[0]
            yshift=x[1]
            shifts=((yshift,0),)*n_proj + base_shifts
            return(a,shifts)

        x0=params_to_vec(self.camera_rotation, self.shifts_current)

        def fun(x):
            "cost function. entropy at multiple slices"
            
            # calculate new alignmnt values
            a,shifts=vec_to_params(x)
            
            # define reconstruction with specific shifts and camera rotation
            reconstructor=self.get_reconstructor(camera_rotation=a, shifts=shifts)
            rec_sub=reconstructor.reconstruct()
            rec_sub=reconstructor.mask_reconstruction_circle(rec_sub, circle_mask_edge)

            costs=[]
            for i in indices:
                slice=np.average(rec_sub[:,:,i:i+thickness],axis=2) #np.sum(rec_sub[:,:,i:i+thickness],axis=2) rec_sub[:,:,i]
                c=self.costfunction_entropy(slice, hist_min=hist_min, hist_max=hist_max, bins=n_bins)
                costs.append(c)
            cost=np.sum(costs)
            return cost  # cf(sub_vol)

        if plot_input:
            # calculate new alignmnt values
            a,shifts=vec_to_params(x0)
            
            # define reconstruction with specific shifts and camera rotation
            reconstructor=self.get_reconstructor(camera_rotation=a, shifts=shifts)
            rec_sub=reconstructor.reconstruct()
            rec_sub=reconstructor.mask_reconstruction_circle(rec_sub, circle_mask_edge)

            costs=[]
            slices=[]
            labels=[]
            for i in indices:
                slice=np.average(rec_sub[:,:,i:i+thickness],axis=2) #np.sum(rec_sub[:,:,i:i+thickness],axis=2) rec_sub[:,:,i]
                c=self.costfunction_entropy(slice, hist_min=hist_min, hist_max=hist_max, bins=n_bins)
                costs.append(c)
                slices.append(slice)
                labels.append(f"slice at x = {i}. Slice cost = {c:.3e}")
            subplots_n(slices, labels=labels, title="Initial slices")
        
        return CostFunction(fun,  bounds, x0, params_to_vec, vec_to_params)


    def get_costfunction_shifts(self, plot_input=False):
        """
        Cost function for maximising the physical variance of a sharp feature in projections
        
        returns (cost_fun, bounds, x0, vectorise_params)
        """
        r=self.get_reconstructor()
        test_rec=r.reconstruct()
        volume_slice=self.get_volume_slice(test_rec) #for making sub-volumes
        n_proj=len(self.projection_x_ang)

        def params_to_vec(shifts):
            sy=[s[0] for s in shifts]
            sx=[s[1] for s in shifts]
            x=sy+sx
            return tuple(x) #make immutable for safety consistency
            
        def vec_to_params(x):
            sy=x[:n_proj]
            sx=x[n_proj:]
            shifts=tuple(zip(sy,sx))
            return(shifts) 

        x0=params_to_vec(self.shifts_current)

        #maxium shift is 1/5 image width (considering extension when rotated), or at least one pixel
        bound_side=np.max((1,np.max(test_rec.shape)//5)) 
        bounds=[(s-bound_side, s+bound_side) for s in x0] # bounds for shifts

        def fun(x): 
            "cost function. fun(x) == sub_volume variance with given shifts"
            
            # calculate new geometry
            shifts=vec_to_params(x)
            
            # apply the geometry to calculate the reconstruction
            reconstructor=self.get_reconstructor(camera_rotation=self.camera_rotation, shifts=shifts) 
            rec_sub=reconstructor.reconstruct()
            
            sub_vol = rec_sub[volume_slice]

            c=self.costfunction_variance(sub_vol)
            return c


        if plot_input:
            # calculate new alignment
            shifts=vec_to_params(x0)
            
            # apply the geometry to calculate the reconstruction
            reconstructor=self.get_reconstructor(camera_rotation=self.camera_rotation, shifts=shifts) 
            rec_sub=reconstructor.reconstruct()
            
            sub_vol = rec_sub[volume_slice]

            c=self.costfunction_variance(sub_vol)
            reconstructor.plot_reconstruction(sub_vol, title="Initial Sub Volume. Cost = %.3e"%fun(x0))
        
        return CostFunction(fun,  bounds, x0, params_to_vec, vec_to_params)

    
    def get_costfunction_shifts_centered(self, plot_input=False):
        """
        Cost function for maximising the physical variance of a sharp feature in projections
        Includes a correction that makes the mean shift stay centered. 
        returns (cost_fun, bounds, x0, vectorise_params)
        """

        cf=self.get_costfunction_shifts(plot_input=plot_input)
        
        n_proj=len(self.projection_x_ang)
        centre_shift=np.mean(self.shifts_current,axis=0)

        #only need to redefine parameter vector to parameter datastructure conversion 
        def vec_to_params(x):
            sy=x[:n_proj]
            sx=x[n_proj:]
            shifts=tuple(zip(sy,sx))
            shifts=np.array(shifts)-np.mean(shifts,axis=0)+centre_shift #correction preveting centre drift
            return(shifts) 

        cf.vec_to_params=vec_to_params
        
        return cf
    
    
    def get_costfuction_rotation_and_shifts(self, plot_input=False):
        """
        Function for maximising the variance of a sharp 3D feature. Varies both tilt axis and shifts.
        
        Returns CostFunction object, containing (cost_fun, bounds, x0, vec_to_params, params_to_vec)
        """
        
        r=self.get_reconstructor()
        test_rec=r.reconstruct()
        volume_slice=self.get_volume_slice(test_rec) #for making sub-volumes
        n_proj=len(self.projection_x_ang)
        
        def params_to_vec(camera_rot, shifts):
                sy=[s[0] for s in shifts]
                sx=[s[1] for s in shifts]
                x=[camera_rot]+sy+sx
                return tuple(x) #make immutable for safety consistency
            
        def vec_to_params(x):
            camera_rot=x[0]
            x=x[1:]
            sy=x[:n_proj]
            sx=x[n_proj:]
            shifts=np.array(tuple(zip(sy,sx)))
            return(camera_rot, shifts) 

        x0=params_to_vec(self.camera_rotation, self.shifts_current)

        #maxium shift is 1/5 image width (considering extension when rotated), or at least one pixel
        bound_side=np.max((1,np.max(test_rec.shape)//5)) 
        bounds=[(self.camera_rotation-10,self.camera_rotation+10),]+[(s-bound_side, s+bound_side) for s in x0[1:]] #centered on x0
        
        def fun(x): 
            "cost function. fun(x) == sub_volume variance with given shifts and camera rotation"
            
            # calculate new alignment
            camera_rot, shifts=vec_to_params(x)
            
            # apply the geometry to calculate the reconstruction
            reconstructor=self.get_reconstructor(camera_rotation=camera_rot, shifts=shifts) 
            rec_sub=reconstructor.reconstruct()
            
            sub_vol = rec_sub[volume_slice]

            c=self.costfunction_variance(sub_vol)
            return c

                
        if plot_input:
            # calculate new alignment
            camera_rot, shifts=vec_to_params(x0)
            
            # apply the geometry to calculate the reconstruction
            reconstructor=self.get_reconstructor(camera_rotation=camera_rot, shifts=shifts) 
            rec_sub=reconstructor.reconstruct()
            
            sub_vol = rec_sub[volume_slice]

            c=self.costfunction_variance(sub_vol)
            reconstructor.plot_reconstruction(sub_vol, title="Initial Sub Volume. Cost = %.3e"%fun(x0))
            

        return CostFunction(fun,  bounds, x0, params_to_vec, vec_to_params)

    
    def get_costfuction_rotation_and_shifts_centered(self, plot_input=False):
        """
        Function for maximising the variance of a sharp 3D feature. Varies both tilt axis and shifts.
        With centre shift correction.
        Returns CostFunction object, containing (cost_fun, bounds, x0, vec_to_params, params_to_vec)
        """

        cf=self.get_costfuction_rotation_and_shifts(plot_input=plot_input)
        
        n_proj=len(self.projection_x_ang)
        centre_shift=np.mean(self.shifts_current,axis=0)
            
        def vec_to_params(x):
            camera_rot=x[0]
            x=x[1:]
            sy=x[:n_proj]
            sx=x[n_proj:]
            shifts=tuple(zip(sy,sx))
            shifts=np.array(shifts)-np.mean(shifts,axis=0)+centre_shift #centre shift correction
            return(camera_rot, shifts) 

        cf.vec_to_params=vec_to_params
            
        return cf
