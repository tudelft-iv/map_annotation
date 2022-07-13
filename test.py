
import numpy
import scipy.interpolate
import matplotlib.pyplot as plt

path_x = numpy.asarray((4.0, 5.638304088577984, 6.785456961280076, 5.638304088577984, 4.0),dtype=float)
path_y = numpy.asarray((0.0, 1.147152872702092, 2.7854569612800755, 4.423761049858059, 3.2766081771559668),dtype=float)

# defining arbitrary parameter to parameterize the curve
path_t = numpy.linspace(0,1,path_x.size)
print(path_t.shape)

# this is the position vector with
# x coord (1st row) given by path_x, and
# y coord (2nd row) given by path_y
r = numpy.vstack((path_x.reshape((1,path_x.size)),path_y.reshape((1,path_y.size))))
print(r.shape)

# creating the spline object
spline = scipy.interpolate.interp1d(path_t,r,kind='cubic')

# defining values of the arbitrary parameter over which
# you want to interpolate x and y
# it MUST be within 0 and 1, since you defined
# the spline between path_t=0 and path_t=1
t = numpy.linspace(numpy.min(path_t),numpy.max(path_t),100)

# interpolating along t
# r[0,:] -> interpolated x coordinates
# r[1,:] -> interpolated y coordinates
r = spline(t)

plt.plot(path_x,path_y,'or')
plt.plot(r[0,:],r[1,:],'-k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

