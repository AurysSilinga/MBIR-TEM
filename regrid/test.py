from regrid import regrid
from pylab import *


# semi irregular grid:
data = zeros((10**3,6))
o = mgrid[0:10, 0:10, 0:10]
for i in range(3):
    data[:, i] = o[i].reshape(-1)
data += rand(1000,6)
data -= 0.5


# dummy data
data[:, 3] = sin(data[:, 0] * pi / 2.) + cos(data[:, 1] * pi / 4.) + data[:, 2]
data[:, 4] = data[:, 0] + data[:, 1] + data[:, 2]
data[:, 5] = 0

# initialize regridding
functor = regrid(data)

# regular grid
n = 50
p = mgrid[0:n, 0:n, 0:n] / 5.
data2 = zeros((n ** 3, 6))
for i in range(3):
    data2[:, i] = p[i].reshape(-1)

# perform regridding
functor(data2)

# visualize some portion
data2 = ma.masked_where(isnan(data2), data2)
print data2.shape, data2[:, 3:].shape
pcolor(data2[:, 3].reshape(n, n, n)[:, :, 5])
colorbar()
show()


