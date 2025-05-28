# -*- coding: utf-8 -*-
# Aurys Silinga, 2024
#

"""
Magnetisation reconstruction functions for pyramid(by)AS
"""

import matplotlib.pyplot as plt
import numpy as np
import pyramid as pr
from jutil.norms import _BaseNorm

from .util import *
from .simulation import simulate_reconstruction 
from .alignment import find_edges, make_projection_data
from .reconstructionCUDA import *

def reconstruct_from_phasemaps_simple(data, lam=1e-3, max_iter=100, ramp_order=1, 
                               verbose=True, plot_input=False, plot_results=False, b_0 = 1, **kwargs):
    """
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

    fwd_model = pr.ForwardModel(data, ramp_order=ramp_order) #define a forward model
    reg = pr.FirstOrderRegularisator(data.mask, lam, add_params=fwd_model.ramp.n) #define the regularisator order
    cost = pr.Costfunction(fwd_model, reg) #define the cost function
    # Reconstruct
    magdata_rec = pr.reconstruction.optimize_linear(cost, max_iter=max_iter, verbose=verbose)
    # Finalize ForwardModel (returns workers if multicore):
    fwd_model.finalize()

    # Plot input:
    if plot_input:
            temp=simulate_reconstruction(data, magdata_rec, cost, b_0=b_0)
            
    # Plot results:
    if plot_results:
        mag_ang_plot=pr.VectorData(magdata_rec.a, magdata_rec.field)
        mag_ang_plot = np.sign(mag_ang_plot.field)  * np.sqrt(np.abs(mag_ang_plot))
        mag_ang_plot.plot_quiver3d('Reconstructed Distribution (angle)')
        magdata_rec.plot_quiver3d('Reconstructed Distribution (amplitude)', coloring='amplitude')
        matshow_n([magdata_rec.field_amp[data.dim[0]//2,:,:], np.sum(magdata_rec.field[0,:,:,:],axis=0)],
                  ["magnitude in z slice", "M_x sum along z"], origin='lower')
        
    return(magdata_rec, cost)


class AmplitudeNorm(_BaseNorm):
    r"""
    For 3D vector field this is :math: '\sum_i (amplitude_i - amplitude_mean)^2'
    Also provides the derivative and second derivative for each point in flattened magnetisation distribution
    """
    
    def __init__(self, mask_vec, machine_precision=1e-15):
        self._mask_vec = np.logical_not(mask_vec) #selected items are corrected
        self._precision = machine_precision
        
    def __call__(self, x_vec):
        x,y,z = x_vec.reshape((3, x_vec.shape[0]//3))
        amp = np.sqrt(x*x + y*y + z*z)
        relevant_amp = amp[self._mask_vec] #keep non shell values
        mean = np.mean(relevant_amp)
        err = relevant_amp - mean
        norm = np.dot(err, err)
        return norm

    def jac(self, x_vec):
        x,y,z = x_vec.reshape((3, x_vec.shape[0]//3))
        amp = np.sqrt(x*x + y*y + z*z)
        relevant_amp = amp[self._mask_vec]
        mean = np.mean(relevant_amp)
        
        #find derivatives for each data point
        amp3 = np.tile(amp,3)
        vec1 = 2*(amp3 - mean)
        amp3[amp3<self._precision]=self._precision #prevent division by 0
        vec2=2 * x_vec/amp3
        derivative = vec1 * vec2
        mask3 = np.tile(np.logical_not(self._mask_vec), 3)
        derivative[mask3]=0 #remove points outside mask
        return derivative

    def hess_diag(self, x_vec):
        x,y,z = x_vec.reshape((3, x_vec.shape[0]//3))
        amp = np.sqrt(x*x + y*y + z*z)
        relevant_amp = amp[self._mask_vec]
        mean = np.mean(relevant_amp)
        
        amp3 = np.tile(amp,3)
        amp3[amp3<self._precision]=self._precision #prevent division by 0
        diag = 2 - 2*mean*(amp3*amp3-x_vec*x_vec)/amp3/amp3/amp3
        mask3 = np.tile(np.logical_not(self._mask_vec), 3) #remove points outside mask
        diag = 2*np.ones_like(x_vec) # simplification
        diag[mask3]=0
        return diag # 
    
    def hess_dot(self, x, vec):
        return (self.hess_diag(vec) * vec)    
    
    
class AmplitudeMeanNorm(_BaseNorm):
    r"""
    For 3D vector field this is :math: '\sum_i (amplitude_i - amplitude_mean)^2'
    Also provides the derivative and second derivative for each point in flattened magnetisation distribution
    """
    
    def __init__(self, mean, mask_vec3, machine_precision=1e-15):
        self._mean = mean
        self._mask_vec3 = mask_vec3
        self._precision = machine_precision
        
    def __call__(self, x_vec):
        x_vec=x_vec[np.logical_not(self._mask_vec3)] #remove shell voxels
        x,y,z = x_vec.reshape((3, x_vec.shape[0]//3))
        amp = np.sqrt(x*x + y*y + z*z)
        mean = self._mean
        err = amp - mean
        norm = np.dot(err, err)
        return norm

    def jac(self, x_vec):
        x,y,z = x_vec.reshape((3, x_vec.shape[0]//3))
        amp = np.sqrt(x*x + y*y + z*z)
        mean = self._mean
        
        #find derivatives for each data point
        amp3 = np.tile(amp,3)
        vec1 = 2*(amp3 - mean)
        amp3[amp3<self._precision]=self._precision #prevent division by 0
        vec2=2 * x_vec/amp3
        derivative = vec1 * vec2
        derivative[self._mask_vec3]=0 #remove shell contribution
        return derivative

    def hess_diag(self, x_vec):
        diag= 2*np.ones_like(x_vec)
        diag[self._mask_vec3]=0
        return diag
    
    def hess_dot(self, x, vec):
        return (self.hess_diag(vec) * vec)
    

class AmplitudeRegulariser(pr.Regularisator):
    """Class for providing a regularisation term which implements Lp norm minimization.

    The constraint this class represents is the minimization of vector Amplitude from mean Amplitude within given mask.

    Attributes
    ----------
    lam: float
        Regularisation parameter determining the weighting between measurements and regularisation.
    add_params : int
        Number of additional parameters which are not used in the regularisation. Used to cut
        the input vector into the appropriate size.
    """
    
    def __init__(self, data_mask=None, reg_mask = None, lam=1e-4, add_params=0):
        
        #define mask that shows which voxels should be regularised
        if reg_mask is None:
            mask_vec=data_mask[data_mask]
        else:
            mask_vec = reg_mask[data_mask] # vector =  reg_mask in positions where data_mask==True
            
        #convert to mask that selects wrong voxels
        mask_vec=np.logical_not(mask_vec)
            
        norm=AmplitudeNorm(mask_vec)
        super().__init__(norm, lam, add_params)
        self._log.debug('Created ' + str(self))
        
        
class AmplitudeMeanRegulariser(pr.Regularisator):
    """Class for providing a regularisation term which implements Lp norm minimization.

    The constraint this class represents is the minimization of vector Amplitude from mean Amplitude within given mask.

    Attributes
    ----------
    lam: float
        Regularisation parameter determining the weighting between measurements and regularisation.
    add_params : int
        Number of additional parameters which are not used in the regularisation. Used to cut
        the input vector into the appropriate size.
    """
    
    def __init__(self, mean=0, data_mask=None, reg_mask = None, lam=1e-4, add_params=0):
        
        #define mask that shows which voxels should not be regularised
        if reg_mask is None:
            mask_vec3=True
        else:
            mask_vec = reg_mask[data_mask]
            mask_vec3 = np.tile(mask_vec, 3)
            
        #convert to mask that selects wrong voxels
        mask_vec3=np.logical_not(mask_vec3)
            
        norm=AmplitudeMeanNorm(mean, mask_vec3)
        
        super().__init__(norm, lam, add_params)
        self._log.debug('Created ' + str(self))
        
        print("mean is",mean)
        
        
        
class ExchangeNorm(_BaseNorm):
    r"""
    For 3D vector field this is the 3D gradient, 
    calculated for x coordinate as (x_i+1 - x_i-1)/2 or (x_i+1-x)/1 at the edges.
    Also provides the second and third derivatives for each point in flattened magnetisation distribution
    """
    
    def __init__(self, diff_vector, scaling, machine_precision=None):
        self._diff_vector = diff_vector
        self._scaling = scaling
        if machine_precision is None:
            self._precision = np.finfo(scaling.dtype).resolution
        
    def __call__(self, x_vec):
        derivatives = self.get_derivatives(x_vec)
        norm = 0
        for axis in derivatives:
            deriv_low, deriv_high = axis
            deriv = deriv_high + deriv_low #scaling is already applied
            norm += np.dot(deriv,deriv)
            
        return norm

    def jac(self, x_vec):
        derivatives = self.get_derivatives(x_vec)
        jacobian = np.zeros(x_vec.shape)
        for axis in derivatives:
            deriv_low, deriv_high = axis
            jacobian += 2*(deriv_low - deriv_high)
            
        return jacobian*(-1) #correction to match jutil

    def hess_diag(self, x_vec):
        diag= 2*np.ones_like(x_vec)
        return diag
    
    def hess_dot(self, x, vec):
        return self.jac(vec) #correction to match jutil
    
    def get_derivatives(self, x_vec):
        derivatives=[]
        for axis in self._diff_vector:
            deriv_low = np.zeros(x_vec.shape)
            deriv_high = np.zeros(x_vec.shape)
            mask_low, mask_high = axis
            
            # deriv_high = x_i+1 - x_i, if x_i+1 exists, else 0
            deriv_high[mask_high] = x_vec[mask_high] - x_vec[mask_low] 
            # deriv_low = x_i - x_i-1, if x_i-1 exists, else 0
            deriv_low[mask_low] = x_vec[mask_high] - x_vec[mask_low] 
            
            derivatives.append((deriv_low/self._scaling, deriv_high/self._scaling))
        return (derivatives)
        
        
class ExchangeRegulariser(pr.Regularisator):
    """Class for providing a regularisation term which implements Lp norm minimization.

    The constraint this class represents is the minimization of Heisenberg exchange energy, 
    with correction for number of nearest neighbors.

    Attributes
    ----------
    lam: float
        Regularisation parameter determining the weighting between measurements and regularisation.
    add_params : int
        Number of additional parameters which are not used in the regularisation. Used to cut
        the input vector into the appropriate size.
    """
    
    def __init__(self, data_mask=None, lam=1e-4, add_params=0):
        
        #create element selector for 3D gradient calculator
        mask=np.pad(data_mask, 1, constant_values=False)
        mxh=mask[1:-1, 1:-1, :-2] [data_mask] #selected elements have value on the left
        mxl=mask[1:-1, 1:-1, 2:]  [data_mask] #has value on the right
        myh=mask[1:-1, :-2, 1:-1] [data_mask]
        myl=mask[1:-1, 2:, 1:-1]  [data_mask]
        mzh=mask[:-2, 1:-1, 1:-1] [data_mask]
        mzl=mask[2:, 1:-1, 1:-1]  [data_mask]
        diff_vector= [[mzl,mzh],[myl,myh],[mxl,mxh]]
        diff_vector = np.tile(diff_vector,(1,1,3)) #tile to account for 3 vector components
        
        #count number of neighbours for each element
        scaling=np.zeros(diff_vector[0,0,:].shape)
        for axis in diff_vector:
            ml, mh = axis
            scaling[ml]=scaling[ml]+1
            scaling[mh]=scaling[mh]+1
        #scaling=np.ones(diff_vector[0,0,:].shape) # to remove scaling correction
        
        norm=ExchangeNorm(diff_vector, scaling)
        
        # lam = 6*lam to be quivalent to base Pyramid regularisator
        super().__init__(norm, lam*6, add_params)
        self._log.debug('Created ' + str(self))
        
        
def reconstruct_from_phasemaps(data, lam=1e-3, max_iter=100, ramp_order=1, 
                               verbose=True, plot_input=True, plot_results=True, b_0 = 1, 
                                   regulariser_type='exchange', mean=None, abs_tol=1e-20, rel_tol=1e-20,
                                   mag_0=None, reg_mask=None):
    """
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

    fwd_model = pr.ForwardModel(data, ramp_order=ramp_order) #define a forward model. How are ramps implemented?
    if regulariser_type == 'amplitude':
        lam1,lam2 = lam
        reg1 = AmplitudeRegulariser(data_mask=data.mask, reg_mask=reg_mask, lam=lam1, add_params=fwd_model.ramp.n)
        reg2 = ExchangeRegulariser(data_mask=data.mask, lam=lam2, add_params=fwd_model.ramp.n)
        reg = pr.ComboRegularisator([reg1, reg2])
        if verbose:
            print("Regularising amplitude and exchange energy")
    elif regulariser_type == 'mean only':
        reg = AmplitudeMeanRegulariser(mean=mean, data_mask=data.mask, reg_mask=reg_mask, lam=lam, add_params=fwd_model.ramp.n)
        if verbose:
            print("Regularising set mean")
    elif regulariser_type == 'amplitude only':
        reg = AmplitudeRegulariser(data_mask=data.mask, reg_mask=reg_mask, lam=lam, add_params=fwd_model.ramp.n)
        if verbose:
            print("Regularising amplitude only")
    elif regulariser_type == 'mean':
        lam1,lam2 = lam
        reg1 = AmplitudeMeanRegulariser(mean=mean, data_mask=data.mask, reg_mask=reg_mask, lam=lam1, add_params=fwd_model.ramp.n)
        reg2 = ExchangeRegulariser(data_mask=data.mask, lam=lam2, add_params=fwd_model.ramp.n)
        reg=pr.ComboRegularisator([reg1, reg2])
        if verbose:
            print("Regularising set mean amplitude and exchange energy")
    elif regulariser_type == 'exchange':
        reg = ExchangeRegulariser(data_mask=data.mask, lam=lam, add_params=fwd_model.ramp.n)
        if verbose:
            print("Regularising exchange energy")
    else:
        reg = pr.FirstOrderRegularisator(data.mask, lam=lam, add_params=fwd_model.ramp.n) #define the regularisator order
        if verbose:
            print("Regularising nearest neighbour difference squared")
    cost = pr.Costfunction(fwd_model, reg, track_cost_iterations=1) #define the cost function
    # Reconstruct and save:
    magdata_rec = pr.reconstruction.optimize_linear(cost, max_iter=max_iter, verbose=verbose, 
                                                    abs_tol=abs_tol, rel_tol=rel_tol, mag_0=mag_0)
    # Finalize ForwardModel (returns workers if multicore):
    fwd_model.finalize()

    # Plot input:
    if plot_input:
            temp=simulate_reconstruction(data, magdata_rec, cost, b_0=b_0)
            
    # Plot results:
    if plot_results:
        mag_ang_plot=pr.VectorData(magdata_rec.a, magdata_rec.field)
        mag_ang_plot = np.sign(mag_ang_plot.field)  * np.sqrt(np.abs(mag_ang_plot))
        mag_ang_plot.plot_quiver3d('Reconstructed Distribution (angle)')
        magdata_rec.plot_quiver3d('Reconstructed Distribution (amplitude)', coloring='amplitude')
        matshow_n([magdata_rec.field_amp[data.dim[0]//2,:,:], np.sum(magdata_rec.field[0,:,:,:],axis=0)],
                  ["magnitude in z slice", "M_x sum along z"], origin='lower')
                  
        
    return(magdata_rec, cost)
    
    
def inspect_magdata(magdata_rec, plot_angles=True, ar_dens=1, mode='arrow'):
    
    magdata_rec.plot_quiver3d('Reconstructed Distribution (amplitude)', coloring='amplitude', ar_dens=ar_dens, mode=mode)
    matshow_n([magdata_rec.field_amp[magdata_rec.dim[0]//2,:,:]],
              ["magnitude in z slice"],origin='lower')
    if plot_angles:
        matshow_n([np.sum(magdata_rec.field[0,:,:,:],axis=0)],
                  ["M_x sum along z"],origin='lower')
        magdata_rec.plot_quiver3d('Reconstructed Distribution (angle)', ar_dens=ar_dens, mode=mode)

        print("Max spin angle:",np.max(get_max_ang(magdata_rec.field, )))

        max_ang = get_max_ang(magdata_rec.field)
        max_ang_field = pr.ScalarData(magdata_rec.a, max_ang)
        max_ang_field.plot_field(title = "Max spin angle")
        return max_ang_field
        
def inspect_cost_values(cost_values, print_chis=False, scale='log'): 
    """
    Plots cost values on a logarithmic graph.
    Can display all the cost values if 'print_chis=True'
    """
    cost_values=np.array(cost_values)
    model_costs = cost_values[:,0]
    regulariser_costs = cost_values[:,1]
    plt.figure()
    ax=plt.gca()
    plt.plot(model_costs,'c.',label="model")
    plt.plot(regulariser_costs,'r.',label="regulariser")
    plt.plot(regulariser_costs+model_costs,'k.', label="sum")
    if scale=='log':
        ax.set_yscale("log")
    plt.title("Reconstruction costfunction values")
    plt.legend()
    plt.show()
    
    if print_chis:
        print()
        print("N =", len(model_costs), "cost value pairs.")
        print("model, regulariser, sum")
        for i in range(len(model_costs)):
            print("%10.5e; %10.5e; %10.5e"%(model_costs[i], regulariser_costs[i], model_costs[i]+regulariser_costs[i]))
        print()

def append_valid_costfunction_values (cost_list, cost_function):
    """
    Takes the ending cost function values of a reconstruction run and appends them to 'cost_list'.
    Acts in place.
    cost_list == [....] goes to
    cost_list = [....., (model_cost_start, regulariser_cost_start), (model_cost_end, regulariser_cost_end)]
    """
    #model_c = cost_function.chisq_m[0]
    # = cost_function.chisq_a[0]
    #cost_list.append((model_c, regulariser_c))
    model_c = cost_function.chisq_m[-1]
    regulariser_c = cost_function.chisq_a[-1]
    cost_list.append((model_c, regulariser_c))


def translate_trim_data_series(data_series, auto_centre=True, x_extension = 0, 
                               last_valid_x_slice = None, tip_x_position = None,  
                               free_space_y_width = 0, free_space_z_width = 0,
                            z_shift=0, y_shift=0, plot_results=False, subcount=5): 

    """
    move the mask to the improved position, trim empty space, add a region for edge moments, and recalculate the projectors.
    If 'autocentre = True' measures the extent of the mask and centres it to remove as much free space as possible.
    Shifts are implemented as reductions to cropping and will fail 
    if they would translate outside the space defined by the mask.
    """
    
    data = data_series
    dz, dy, dx = data.dim
    
    if tip_x_position is None:
        tip_x_position=dx-1
        
    
    if auto_centre: #find the edge most pixels on each axis and centre the mask
        wrongs_3d=np.bitwise_not(data.mask)
        centre_params=[]
        for axis in range(wrongs_3d.ndim):
            low_i, high_i = find_edges(wrongs_3d, axis=axis)
            high_dist=wrongs_3d.shape[axis] - high_i - 1
            if axis==2: #overwrite automatic detection for x-axis to manually correct if necessary
                if last_valid_x_slice is not None: 
                    low_i=last_valid_x_slice
                if tip_x_position is not None:
                    high_i=tip_x_position
            centre_offset = (high_dist - low_i)//2
            free_space_width = int(np.min((low_i,high_dist)) + np.abs(centre_offset))
            centre_params.append((centre_offset, free_space_width))    
        z_shift, free_space_z_width = centre_params[0]
        y_shift, free_space_y_width = centre_params[1]
        x_shift, free_space_x_width = centre_params[2]
        print("Automatic detection:\n[(z_shift, z_crop),(y_shift, y_crop)]\n", centre_params[:2])
    else:
        high_dist_x=dx - 1 - tip_x_position
        x_shift = (high_dist_x - last_valid_x_slice)//2
        free_space_x_width = int(np.min((last_valid_x_slice,high_dist_x)) + x_shift)
        
    #make total dimensional change even:
    if x_extension%2==1:
        x_extension+=1

    #reducing dimension removes pixels form high end
    dim = (dz-free_space_z_width*2, dy-free_space_y_width*2, dx-free_space_x_width*2+x_extension) #change in dimensions 
    mz, my, mx = dim
    #the translation is implementated by translating the mask and cropping all sides evenly
    center = (mz/2+z_shift, my/2+y_shift, mx/2+x_shift + x_extension//2) 
        
    #recalculate the projectors
    zrots=[]
    xtilts=[]
    camera_rots=[]
    centers=[]
    phasemaps = data.phasemaps
    for projector in data.projectors:
        zrots.append(np.degrees(projector.rotation))
        xtilts.append(np.degrees(projector.tilt))
        camera_rots.append(np.degrees(projector.camera_rotation))
        center_old = projector.center
        center_new=[]
        for i in range(3): #center_new =  center_global + old_center_shift
            center_new.append(center[i] + center_old[i] - data.dim[i]/2)
        centers.append(tuple(center_new))
        
    
    data_e =  make_projection_data(phasemaps, zrots, xtilts, camera_rots, data.a, center = centers, dim=dim, 
                                     plot_results=False, save_data=False, subcount=subcount)
    
    #reshape the original mask 
    dz, dy, dx = data_e.dim
    mask0_e = np.pad(data.mask, ((0,0), (0,0),(x_extension, 0)), mode='edge')
    Dz, Dy, Dx = mask0_e.shape

    x_crop = free_space_x_width -x_shift
    mx_crop = -(free_space_x_width) -x_shift
    if mx_crop == 0:
        mx_crop=None
    z_crop = free_space_z_width -z_shift
    mz_crop = -(free_space_z_width) -z_shift
    if mz_crop == 0:
        mz_crop = None
    y_crop = free_space_y_width -y_shift
    my_crop = -(free_space_y_width) -y_shift
    if my_crop == 0:
        my_crop = None
    
    mask0_e = mask0_e[z_crop:mz_crop, y_crop:my_crop, x_crop:mx_crop]
    data_e.mask = mask0_e.copy()

    boundary_charge_edge = x_extension-1
    if boundary_charge_edge>=0:
        #add boundary region
        data_e.mask[:, :, :boundary_charge_edge] = True
        data_e.mask[:, :, boundary_charge_edge] = False

    if plot_results:
        #data.plot_mask(title="original missplaced mask")
        data_e.plot_mask(title="translated mask")
        temp = data_e.mask.copy()
        data_e.set_3d_mask()
        #data_e.plot_mask(title="raw mask calculation with new projectors")
        matshow_n([np.sum(data_e.mask,axis=0),np.sum(temp,axis=0)],["newly calculated mask z sum","translated original mask z sum"])
        data_e.mask=temp
        
    return(data_e, boundary_charge_edge)
        
