# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.Diagnostics` class for the calculation of diagnostics of a
specified costfunction for a fixed magnetization distribution."""

import logging

from pyramid.fielddata import VectorData
from pyramid.phasemap import PhaseMap

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import patheffects
from matplotlib.ticker import FuncFormatter
import numpy as np

import jutil

__all__ = ['Diagnostics', 'get_vector_field_errors']


class Diagnostics(object):
    """Class for calculating diagnostic properties of a specified costfunction.

    For the calculation of diagnostic properties, a costfunction and a magnetization distribution
    are specified at construction. With the :func:`~.set_position`, a position in 3D space can be
    set at which all properties will be calculated. Properties are saved via boolean flags and
    thus, calculation is only done if the position has changed in between. The standard deviation
    and the measurement contribution require the execution of a conjugate gradient solver and can
    take a while for larger problems.

    Attributes
    ----------
    x_rec: :class:`~numpy.ndarray`
        Vectorized magnetization distribution at which the costfunction is evaluated.
    cost: :class:`~.pyramid.costfunction.Costfunction`
        Costfunction for which the diagnostics are calculated.
    max_iter: int, optional
        Maximum number of iterations. Default is 1000.
    fwd_model: :class:`~pyramid.forwardmodel.ForwardModel`
        Forward model used in the costfunction.
    Se_inv : :class:`~numpy.ndarray` (N=2), optional
        Inverted covariance matrix of the measurement errors. The matrix has size `NxN` with N
        being the length of the targetvector y (vectorized phase map information).
    dim: tuple (N=3)
        Dimensions of the 3D magnetic distribution.
    row_idx: int
        Row index of the system matrix corresponding to the current position in 3D space.

    Notes
    -----
        Some properties depend on others, which may require recalculations of these prior
        properties if necessary. The dependencies are ('-->' == 'requires'):
        avrg_kern_row --> gain_row --> std --> m_inv_row
        measure_contribution is independant

    """

    _log = logging.getLogger(__name__ + '.Diagnostics')

    @property
    def cov_row(self):
        """Row of the covariance matrix (``S_a^-1+F'(x_f)^T S_e^-1 F'(x_f)``) which is needed for
        the calculation of the gain and averaging kernel matrizes and which ideally contains the
        variance at position `row_idx` for the current component and position in 3D.
        Note that the covariance matrix of the solution is symmetric (like all covariance
        matrices) and thusly this property could also be called cov_col for column."""
        if not self._updated_cov_row:
            e_i = np.zeros(self.cost.n, dtype=self.x_rec.dtype)
            e_i[self.row_idx] = 1
            row = 2 * jutil.cg.conj_grad_solve(self._A, e_i, P=self._P, max_iter=self.max_iter,
                                               verbose=self.verbose)
            self._std_row = np.asarray(row)
            self._updated_cov_row = True
        return self._std_row

    @property
    def std(self):
        """Standard deviation of the chosen component at the current position (calculated when
        needed)."""
        return np.sqrt(self.cov_row[self.row_idx])

    @property
    def gain_row(self):
        """Row of the gain matrix, which maps differences of phase measurements onto differences in
        the retrieval result of the magnetization distribution(calculated when needed)."""
        if not self._updated_gain_row:
            self._gain_row = self.Se_inv.dot(self.fwd_model.jac_dot(self.x_rec, self.cov_row))
            self._updated_gain_row = True
        return self._gain_row

    @property
    def avrg_kern_row(self):
        """Row of the averaging kernel matrix (which is ideally the identity matrix), which
        describes the smoothing introduced by the regularization (calculated when needed)."""
        if not self._updated_avrg_kern_row:
            self._avrg_kern_row = self.fwd_model.jac_T_dot(self.x_rec, self.gain_row)
            self._updated_avrg_kern_row = True
        return self._avrg_kern_row

    @property
    def measure_contribution(self):
        """The sum over an averaging kernel matrix row, which is an indicator for wheter a point of
        the solution is determined by the measurement (close to `1`) or by a priori information
        (close to `0`)."""
        if not self._updated_measure_contribution:
            cache = self.fwd_model.jac_dot(self.x_rec, np.ones(self.cost.n, self.x_rec.dtype))
            cache = self.fwd_model.jac_T_dot(self.x_rec, self.Se_inv.dot(cache))
            mc = 2 * jutil.cg.conj_grad_solve(self._A, cache, P=self._P, max_iter=self.max_iter)
            self._measure_contribution = mc
            self._updated_measure_contribution = True
        return self._measure_contribution

    @property
    def pos(self):
        """The current solution position, which specifies the 3D-point (and the component) of the
        magnetization, for which diagnostics are calculated."""
        return self._pos

    @pos.setter
    def pos(self, pos):
        c, z, y, x = pos
        assert self.mask[z, y, x], 'Position is outside of the provided mask!'
        mask_vec = self.mask.ravel()
        idx_3d = z * self.dim[1] * self.dim[2] + y * self.dim[2] + x
        row_idx = c * np.prod(mask_vec.sum()) + mask_vec[:idx_3d].sum()
        if row_idx != self.row_idx:
            self._pos = pos
            self.row_idx = row_idx
            self._updated_cov_row = False
            self._updated_gain_row = False
            self._updated_avrg_kern_row = False
            self._updated_measure_contribution = False

    def __init__(self, magdata, cost, max_iter=1000, verbose=False):
        self._log.debug('Calling __init__')
        self.magdata = magdata
        self.cost = cost
        self.max_iter = max_iter
        self.verbose = verbose
        self.fwd_model = cost.fwd_model
        self.Se_inv = cost.Se_inv
        self.dim = cost.fwd_model.data_set.dim
        self.mask = cost.fwd_model.data_set.mask
        self.x_rec = np.empty(cost.n)
        self.x_rec[:self.fwd_model.data_set.n] = self.magdata.get_vector(mask=self.mask)
        self.x_rec[self.fwd_model.data_set.n:] = self.fwd_model.ramp.param_cache.ravel()
        self.row_idx = None
        self.pos = (0,) + tuple(np.array(np.where(self.mask))[:, 0])  # first True mask entry
        self._updated_cov_row = False
        self._updated_gain_row = False
        self._updated_avrg_kern_row = False
        self._updated_measure_contribution = False
        self._A = jutil.operator.CostFunctionOperator(self.cost, self.x_rec)
        self._P = jutil.preconditioner.CostFunctionPreconditioner(self.cost, self.x_rec)
        self._log.debug('Creating ' + str(self))

    def get_avrg_kern_field(self, pos=None):
        """Get the averaging kernel matrix row represented as a 3D magnetization distribution.

        Parameters
        ----------
        pos: tuple (N=4)
            Position in 3D plus component `(c, z, y, x)`

        Returns
        -------
        magdata_avrg_kern: :class:`~pyramid.fielddata.VectorData`
            Averaging kernel matrix row represented as a 3D magnetization distribution

        """
        self._log.debug('Calling get_avrg_kern_field')
        if pos is not None:
            self.pos = pos
        magdata_avrg_kern = VectorData(self.cost.fwd_model.data_set.a, np.zeros((3,) + self.dim))
        vector = self.avrg_kern_row[:-self.fwd_model.ramp.n]  # Only take vector field, not ramp!
        magdata_avrg_kern.set_vector(vector, mask=self.mask)
        return magdata_avrg_kern

    def calculate_fwhm(self, pos=None, plot=False):
        """Calculate and plot the averaging pixel number at a specified position for x, y or z.

        Parameters
        ----------
        pos: tuple (N=4)
            Position in 3D plus component `(c, z, y, x)`
        plot : bool, optional
            If True, a FWHM linescan plot is shown. Default is False.

        Returns
        -------
        fwhm : float
            The FWHM in x, y and z direction. The inverse corresponds to the number of pixels over
            which is approximately averaged.
        lr : 3 tuples of 2 floats
            The left and right borders in x, y and z direction from which the FWHM is calculated.
            Given in pixel coordinates and relative to the current position!
        cxyz_dat : 4 lists of floats
            The slices through the current position in the 4D volume (including the component),
            which were used for FWHM calculations. Denotes information content in %!

        Notes
        -----
        Uses the :func:`~.get_avrg_kern_field` function

        """
        self._log.debug('Calling calculate_fwhm')
        a = self.magdata.a
        magdata_avrg_kern = self.get_avrg_kern_field(pos)
        x = np.arange(0, self.dim[2]) - self.pos[3]
        y = np.arange(0, self.dim[1]) - self.pos[2]
        z = np.arange(0, self.dim[0]) - self.pos[1]
        c_dat = magdata_avrg_kern.field[:, self.pos[1], self.pos[2], self.pos[3]]
        x_dat = magdata_avrg_kern.field[self.pos[0], self.pos[1], self.pos[2], :]
        y_dat = magdata_avrg_kern.field[self.pos[0], self.pos[1], :, self.pos[3]]
        z_dat = magdata_avrg_kern.field[self.pos[0], :, self.pos[2], self.pos[3]]
        c_dat = np.asarray(c_dat * 100)  # in %
        x_dat = np.asarray(x_dat * 100)  # in %
        y_dat = np.asarray(y_dat * 100)  # in %
        z_dat = np.asarray(z_dat * 100)  # in %

        def _calc_lr(c):
            data = [x_dat, y_dat, z_dat][c]
            i_m = np.argmax(data)  # Index of the maximum
            # Left side:
            l = i_m
            for i in np.arange(i_m - 1, -1, -1):
                if data[i] < data[i_m] / 2:
                    # Linear interpolation between i and i + 1 to find left fractional index pos:
                    l = (data[i_m] / 2 - data[i]) / (data[i + 1] - data[i]) + i
                    break
            # Right side:
            r = i_m
            for i in np.arange(i_m + 1, data.size):
                if data[i] < data[i_m] / 2:
                    # Linear interpolation between i and i - 1 to find right fractional index pos:
                    r = (data[i_m] / 2 - data[i - 1]) / (data[i] - data[i - 1]) + i - 1
                    break
            # Transform from index to coordinates:
            l = (l - self.pos[3-c])
            r = (r - self.pos[3-c])
            return l, r

        # Calculate FWHM:
        lx, rx = _calc_lr(0)
        ly, ry = _calc_lr(1)
        lz, rz = _calc_lr(2)

        # TODO: Test if FWHM is really calculated with a in mind... didn't seem so...
        fwhm_x = (rx - lx) * a
        fwhm_y = (ry - ly) * a
        fwhm_z = (rz - lz) * a
        # Plot helpful stuff:
        if plot:
            fig, axis = plt.subplots(1, 1)
            axis.axvline(x=0, ls='-', color='k', linewidth=2)
            axis.axhline(y=0, ls='-', color='k', linewidth=2)
            axis.axhline(y=x_dat.max(), ls='-', color='k', linewidth=2)
            axis.axhline(y=x_dat.max() / 2, ls='--', color='k', linewidth=2)
            axis.vlines(x=[lx, rx], ymin=0, ymax=x_dat.max() / 2, linestyles='--',
                        color='r', linewidth=2, alpha=0.5)
            axis.vlines(x=[ly, ry], ymin=0, ymax=y_dat.max() / 2, linestyles='--',
                        color='g', linewidth=2, alpha=0.5)
            axis.vlines(x=[lz, rz], ymin=0, ymax=z_dat.max() / 2, linestyles='--',
                        color='b', linewidth=2, alpha=0.5)
            l = []
            l.extend(axis.plot(x, x_dat, label='x-dim.', color='r', marker='o', linewidth=2))
            l.extend(axis.plot(y, y_dat, label='y-dim.', color='g', marker='o', linewidth=2))
            l.extend(axis.plot(z, z_dat, label='z-dim.', color='b', marker='o', linewidth=2))
            cx = axis.scatter(0, c_dat[0], marker='o', s=200, edgecolor='r', label='x-comp.',
                              facecolor='r', alpha=0.75)
            cy = axis.scatter(0, c_dat[1], marker='d', s=200, edgecolor='g', label='y-comp.',
                              facecolor='g', alpha=0.75)
            cz = axis.scatter(0, c_dat[2], marker='*', s=200, edgecolor='b', label='z-comp.',
                              facecolor='b', alpha=0.75)
            lim_min = np.min(np.concatenate((x, y, z))) - 0.5
            lim_max = np.max(np.concatenate((x, y, z))) + 0.5
            axis.set_xlim(lim_min, lim_max)
            axis.set_title('Avrg. kern. FWHM', fontsize=18)
            axis.set_xlabel('x/y/z-slice [nm]', fontsize=15)
            axis.set_ylabel('information content [%]', fontsize=15)
            axis.tick_params(axis='both', which='major', labelsize=14)
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.3g}'.format(x * a)))
            comp_legend = axis.legend([cx, cy, cz], [c.get_label() for c in [cx, cy, cz]], loc=2,
                                      scatterpoints=1, prop={'size': 14})
            axis.legend(l, [i.get_label() for i in l], loc=1, numpoints=1, prop={'size': 14})
            axis.add_artist(comp_legend)
        fwhm = fwhm_x, fwhm_y, fwhm_z
        lr = (lx, rx), (ly, ry), (lz, rz)
        cxyz_dat = c_dat, x_dat, y_dat, z_dat
        return fwhm, lr, cxyz_dat

    def get_gain_maps(self, pos=None):
        """Get the gain matrix row represented as a list of 2D (inverse) phase maps.

        Parameters
        ----------
        pos: tuple (N=4)
            Position in 3D plus component `(c, z, y, x)`

        Returns
        -------
        gain_map_list: list of :class:`~pyramid.phasemap.PhaseMap`
            Gain matrix row represented as a list of 2D phase maps

        Notes
        -----
        Note that the produced gain maps define the magnetization change at the current position
        in 3d per phase change at the position of the . Take this into account when plotting the
        maps (1/rad instead of rad).

        """
        self._log.debug('Calling get_gain_maps')
        if pos is not None:
            self.pos = pos
        hp = self.cost.fwd_model.data_set.hook_points
        gain_map_list = []
        for i, projector in enumerate(self.cost.fwd_model.data_set.projectors):
            gain = self.gain_row[hp[i]:hp[i + 1]].reshape(projector.dim_uv)
            gain_map_list.append(PhaseMap(self.cost.fwd_model.data_set.a, gain))
        return gain_map_list

    def plot_position(self, **kwargs):
        proj_axis = kwargs.get('proj_axis', 'z')
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            pos_2d = (self.pos[2], self.pos[3])
            ax_slice = self.pos[1]
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            pos_2d = (self.pos[1], self.pos[3])
            ax_slice = self.pos[2]
        elif proj_axis == 'x':  # Slice of the zy-plane with x = ax_slice
            pos_2d = (self.pos[2], self.pos[1])
            ax_slice = self.pos[3]
        else:
            raise ValueError('{} is not a valid argument (use x, y or z)'.format(proj_axis))
        note = kwargs.pop('note', None)
        if note is None:
            comp = {0: 'x', 1: 'y', 2: 'z'}[self.pos[0]]
            note = '{}-comp., pos.: {}'.format(comp, self.pos[1:])
        # Plots:
        axis = self.magdata.plot_quiver_field(note=note, ax_slice=ax_slice, **kwargs)
        rect = axis.add_patch(patches.Rectangle((pos_2d[1], pos_2d[0]), 1, 1, fill=False,
                                                edgecolor='w', linewidth=2, alpha=0.5))
        rect.set_path_effects([patheffects.withStroke(linewidth=4, foreground='k', alpha=0.5)])

    def plot_position3d(self, **kwargs):
        pass

    def plot_avrg_kern_field(self, pos=None, **kwargs):
        a = self.magdata.a
        avrg_kern_field = self.get_avrg_kern_field(pos)
        fwhms, lr = self.calculate_fwhm(pos)[:2]
        proj_axis = kwargs.get('proj_axis', 'z')
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            pos_2d = (self.pos[2], self.pos[3])
            ax_slice = self.pos[1]
            width, height = fwhms[0] / a, fwhms[1] / a
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            pos_2d = (self.pos[1], self.pos[3])
            ax_slice = self.pos[2]
            width, height = fwhms[0] / a, fwhms[2] / a
        elif proj_axis == 'x':  # Slice of the zy-plane with x = ax_slice
            pos_2d = (self.pos[2], self.pos[1])
            ax_slice = self.pos[3]
            width, height = fwhms[2] / a, fwhms[1] / a
        else:
            raise ValueError('{} is not a valid argument (use x, y or z)'.format(proj_axis))
        note = kwargs.pop('note', None)
        if note is None:
            comp = {0: 'x', 1: 'y', 2: 'z'}[self.pos[0]]
            note = '{}-comp., pos.: {}'.format(comp, self.pos[1:])
        # Plots:
        axis = avrg_kern_field.plot_quiver_field(note=note, ax_slice=ax_slice, **kwargs)
        xy = (pos_2d[1], pos_2d[0])
        rect = axis.add_patch(patches.Rectangle(xy, 1, 1, fill=False, edgecolor='w',
                                                linewidth=2, alpha=0.5))
        rect.set_path_effects([patheffects.withStroke(linewidth=4, foreground='k', alpha=0.5)])
        xy = (xy[0] + 0.5, xy[1] + 0.5)
        artist = axis.add_patch(patches.Ellipse(xy, width, height, fill=False, edgecolor='w',
                                                linewidth=2, alpha=0.5))
        artist.set_path_effects([patheffects.withStroke(linewidth=4, foreground='k', alpha=0.5)])


    def plot_avrg_kern_field3d(self, pos=None, mask=True, ellipsoid=True, **kwargs):
        avrg_kern_field = self.get_avrg_kern_field(pos)
        avrg_kern_field.plot_mask(color=(1, 1, 1), opacity=0.15, labels=False, grid=False,
                                  orientation=False)
        avrg_kern_field.plot_quiver3d(**kwargs, new_fig=False)
        fwhm = self.calculate_fwhm()[0]
        from mayavi.sources.api import ParametricSurface
        from mayavi.modules.api import Surface
        from mayavi import mlab
        engine = mlab.get_engine()
        scene = engine.scenes[0]
        scene.scene.disable_render = True  # for speed  # TODO: EVERYWHERE WITH MAYAVI!
        # TODO: from enthought.mayavi import mlab
        # TODO: f = mlab.figure() # returns the current scene
        # TODO: engine = mlab.get_engine() # returns the running mayavi engine
        source = ParametricSurface()
        source.function = 'ellipsoid'
        engine.add_source(source)
        surface = Surface()
        source.add_module(surface)

        actor = surface.actor  # mayavi actor, actor.actor is tvtk actor
        # actor.property.ambient = 1 # defaults to 0 for some reason, ah don't need it, turn off scalar visibility instead
        actor.property.opacity = 0.5
        actor.property.color = (0, 0, 0)
        actor.mapper.scalar_visibility = False  # don't colour ellipses by their scalar indices into colour map
        actor.property.backface_culling = True  # gets rid of rendering artifact when opacity is < 1
        # actor.property.frontface_culling = True
        actor.actor.orientation = [0, 0, 0]  # in degrees
        actor.actor.origin = (0, 0, 0)
        actor.actor.position = (self.pos[1]+0.5, self.pos[2]+0.5, self.pos[3]+0.5)
        a = self.magdata.a
        actor.actor.scale = [0.5*fwhm[0]/a, 0.5*fwhm[1]/a, 0.5*fwhm[2]/a]

        #surface.append(surface)


        scene.scene.disable_render = False  # now turn it on  # TODO: EVERYWHERE WITH MAYAVI!


    def plot_avrg_kern_field_3d_to_2d(self, dim_uv=None, axis=None, figsize=None, high_res=False,
                                      **kwargs):
        # TODO: 3d_to_2d into plottools and make available for all 3D plots if possible!
        import tempfile
        from PIL import Image
        import os
        from . import plottools
        from mayavi import mlab
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        if axis is None:
            self._log.debug('axis is None')
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1)
            axis.set_axis_bgcolor('gray')
        kwargs.setdefault('labels', 'False')
        #avrg_kern_field = self.get_avrg_kern_field()
        #avrg_kern_field.plot_quiver3d(**kwargs)
        self.plot_avrg_kern_field3d(**kwargs)
        if high_res:  # Use temp files:
            tmpdir = tempfile.mkdtemp()
            temp_path = os.path.join(tmpdir, 'temp.png')
            try:
                mlab.savefig(temp_path, size=(2000, 2000))
                imgmap = np.asarray(Image.open(temp_path))
            except Exception as e:
                raise e
            finally:
                os.remove(temp_path)
                os.rmdir(tmpdir)
        else:  # Use screenshot (returns array WITH alpha!):
            imgmap = mlab.screenshot(mode='rgba', antialiased=True)
        mlab.close(mlab.gcf())
        if dim_uv is None:
            dim_uv = self.dim[1:]
        axis.imshow(imgmap, extent=[0, dim_uv[0], 0, dim_uv[1]], origin='upper')
        kwargs.setdefault('scalebar', False)
        kwargs.setdefault('hideaxes', True)
        return plottools.format_axis(axis, hideaxes=True, scalebar=False)


def get_vector_field_errors(vector_data, vector_data_ref):
    """After Kemp et. al.: Analysis of noise-induced errors in vector-field electron tomography"""
    v, vr = vector_data.field, vector_data_ref.field
    va, vra = vector_data.field_amp, vector_data_ref.field_amp
    volume = np.prod(vector_data.dim)
    # Total error:
    amp_sum_sqr = np.nansum((v - vr)**2)
    rms_tot = np.sqrt(amp_sum_sqr / np.nansum(vra**2))
    # Directional error:
    scal_prod = np.clip(np.nansum(vr * v, axis=0) / (vra * va), -1, 1)  # arccos float pt. inacc.!
    rms_dir = np.sqrt(np.nansum(np.arccos(scal_prod)**2) / volume) / np.pi
    # Magnitude error:
    rms_mag = np.sqrt(np.nansum((va - vra)**2) / np.nansum(vra**2))
    # Return results as tuple:
    return rms_tot, rms_dir, rms_mag
