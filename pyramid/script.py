# # -*- coding: utf-8 -*-
# # %%
# from PyQt5 import Qt
# import matplotlib.pyplot as plt
# import pyramid as pr

# magexp = pr.load_vectordata(r'C:\Users\caron\Desktop\Diagnostics 1E1\magdata_rec_lam1e+01.hdf5')
# magsim = pr.load_vectordata(r'C:\Users\caron\Desktop\Diagnostics 1E1\magdata_sim.hdf5', a=magexp.a)
# magexp.plot_quiver3d(ar_dens=3)
# magsim.plot_quiver3d(ar_dens=3)

# # %%
# magdata.plot_quiver_field()
# magdata = pr.magcreator.examples.smooth_vortex_disc()
# # %%
# plt.show()

# # %%
# magdata.plot_quiver3d()


# # import sys, importlib
# # from PyQt5 import QtWidgets

# # print('Sys Path:')
# # print('  %s\n' % '\n  '.join(sys.path))

# # mod = None
# # modname = 'collections.abc'

# # modname = 'PyQt5.Qt'
# # if sys.argv[-1] == '1':
# #     print('Importing Before...\n')

# #     mod = importlib.import_module(modname)
# #     app = QtWidgets.QApplication(sys.argv)

# # elif sys.argv[-1] == '2':
# #     print('Importing After...\n')
# #     app = QtWidgets.QApplication(sys.argv)
# #     mod = importlib.import_module(modname)
# #     from PyQt5 import Qt

# # print('Result: %r' % mod)
# #
# # %% [markdown]
# # # Imports:

# # %% Imports
# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import pyramid as pr

# # %% [markdown]
# # #

# filename = os.path.join(R'c:\Users\caron\Work\Projects\pyramid\pyramid', R'Sith_skyrm_M.tif')
# phasemap_my = pr.load_phase_map(filename=filename)
# phasemap_my.plot_combined()

# # %%
# magdata = pr.magcreator.examples.homog_slab(phi=np.pi / 2, theta=np.pi)
# x = []
# for angle in np.arange(0, 360):
#     x.append(pr.utils.pm(magdata, mode='x-tilt', tilt=np.deg2rad(angle)).phase.max())
# plt.plot(x)

# # %%
# max_iter = 1000
# ramp_order = 1
# dim=(1,)+phasemap.dim_uv
# lambdas = [E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2]

# data_set = pr.DataSet(phasemap.a, dim)
# pr.SimpleProjector()
# data_set.append(phasemap, pr.SimpleProjector(dim))

# fwd_model = pr.ForwardModel(data_set, ramp_order=${:ramp_order})

# lcurve = pr.LCurve(fwd_model, max_iter)
# lcurve.calculate(lambdas)
# lcurve.plot(lambdas)









# %%
import pyramid as pr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %%
magdata = pr.magcreator.examples.smooth_vortex_disc()

# %%
pr.plottools.pretty_plots()
axis = magdata.plot_quiver_field(colorwheel=True)
plt.show()
pass
