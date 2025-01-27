# Copyright 2014 (C) Adam Greig, Daniel Richman
#
# This file is part of Tawhiri.
#
# Tawhiri is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Tawhiri is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Tawhiri.  If not, see <http://www.gnu.org/licenses/>.

# Cython compiler directives:
#
# cython: language_level=3
#
# pick(...) is careful in what it returns:
# cython: boundscheck=False
# cython: wraparound=False
#
# We check for division by zero, and don't divide by negative values
# (unless the dataset is really dodgy!):
# cython: cdivision=True

"""
Interpolation to determine wind velocity at any given time,
latitude, longitude and altitude.

Note that this module is compiled with Cython to enable fast
memory access.
"""


from magicmemoryview import MagicMemoryView
from .warnings cimport WarningCounts

DEF VAR_A = 0
DEF VAR_U = 1
DEF VAR_V = 2
DEF VAR_W = 3


ctypedef float[:, :, :, :, :] dataset

cdef struct Lerp1:
    long index
    double lerp

cdef struct Lerp3:
    long hour, lat, lng
    double lerp


class RangeError(ValueError):
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        s = "{0}={1}".format(variable, value)
        super(RangeError, self).__init__(s)


def make_interpolator(dataset, WarningCounts warnings):
    """
    Produce a function that can get wind data from `dataset`

    This wrapper casts :attr:`Dataset.array` into a form that is useful
    to us, and then returns a closure that can be used to retrieve
    wind velocities.
    """

    cdef float[:, :, :, :, :] data

    if warnings is None:
        raise TypeError("Warnings must not be None")

    # GFS
    #data = MagicMemoryView(dataset.array, (2, 49, 4, 721, 1440), b"f")
    # ECMWF
    #data = MagicMemoryView(dataset.array, (3, 13, 4, 721, 1440), b"f")
    # MEPS
    #data = MagicMemoryView(dataset.array, (4, 65, 4, 88, 28), b"f")
    # Header
    data = MagicMemoryView(dataset.array, (
        dataset.header['shape']['hour']['count'], 
        dataset.header['shape']['levels'], 
        len(dataset.header['shape']['variables']), 
        dataset.header['shape']['x']['count'], 
        dataset.header['shape']['y']['count']), b"f")

    # Grab indexes of variables
    supported_vars = ['A', 'U', 'V', 'W'] # Matching VAR_A, VAR_U, ...
    cdef int[4] vars
    for idx,key in enumerate(supported_vars):
        if key in dataset.header['shape']['variables']:
            vars[idx] = dataset.header['shape']['variables'].index(key)
        else:
            vars[idx] = -1

    def f(hour, lat, lng, alt):
        return get_wind(data, warnings, vars, hour, lat, lng, alt)

    return f


cdef object get_wind(dataset ds, WarningCounts warnings, int[4] vars,
                     double hour, double lat, double lng, double alt):
    """
    Return [u, v] wind components for the given position.
    Time is in fractional hours since the dataset starts.
    Alt is metres above sea level.
    Lat is latitude in decimal degrees, -90 to +90.
    Lng is longitude in decimal degrees, 0 to 360.

    Returned coordinates are interpolated from the surrounding grid
    points in time, latitude, longitude and altitude.
    """

    cdef Lerp3[8] lerps
    cdef long altidx
    cdef double lower, upper, u, v, w

    pick3(ds, hour, lat, lng, lerps)

    altidx = search(ds, lerps, alt)
    lower = interp3(ds, lerps, vars[VAR_A], altidx)
    upper = interp3(ds, lerps, vars[VAR_A], altidx + 1)

    if lower != upper:
        lerp = (upper - alt) / (upper - lower)
    else:
        lerp = 0.5

    if lerp < 0: warnings.altitude_too_high += 1

    cdef Lerp1 alt_lerp = Lerp1(altidx, lerp)

    u = interp4(ds, lerps, alt_lerp, vars[VAR_U])
    v = interp4(ds, lerps, alt_lerp, vars[VAR_V])
    w = interp4(ds, lerps, alt_lerp, vars[VAR_W]) if vars[VAR_W] is not -1 else 0

    print("Wind at hour {}, {},{} @ {} (idx {}, {}/{}={}) = {},{},{}".format(hour, lat, lng, alt, alt_lerp.index, lower, upper, lerp, u, v, w))

    return u, v, w,

cdef long pick(double left, double step, long n, double value,
               object variable_name, Lerp1[2] out) except -1:

    cdef double a, l
    cdef long b

    a = (value - left) / step
    b = <long> a
    if b < 0 or b >= n - 1:
        if variable_name == "hour":
            out[0] = Lerp1(0, 0)
            out[1] = Lerp1(0, 0)
            return 0
        else:
            raise RangeError(variable_name, value)
    l = a - b

    out[0] = Lerp1(b, 1 - l)
    out[1] = Lerp1(b + 1, l)
    return 0

cdef long pick3(dataset ds, double hour, double lat, double lng, Lerp3[8] out) except -1:
    cdef Lerp1[2] lhour, llat, llng

    print(f"pick3: {hour}, {lat}, {lng}")

    # the dimensions of the lat/lon axes are 361 and 720
    # (The latitude axis includes its two endpoints; the longitude only
    # includes the lower endpoint)
    # However, the longitude does wrap around, so we tell `pick` that the
    # longitude axis is one larger than it is (so that it can "choose" the
    # 721st point/the 360 degrees point), then wrap it afterwards.
    # GFS
    #pick(0, 1, 2, hour, "hour", lhour)
    # ECMWF
    #pick(0, 3, 3, hour, "hour", lhour)
    # MEPS
    #pick(0, 1, 4, hour, "hour", lhour)
    # Header
    # TODO: ds is actually the float array – We should make a struct or something instead?
    pick(ds.header['shape']['hour']['min'], ds.header['shape']['hour']['step'], ds.header['shape']['hour']['count'], hour, "hour", lhour)
    
    #pick(-90, 0.25, 721, lat, "lat", llat)
    #pick(-180, 0.25, 1440 + 1, lng, "lng", llng)
    #pick(0, 0.25, 1440 + 1, lng, "lng", llng)
    #if llng[1].index == 361:
    #    raise Exception("Edge case")
    #    llng[1].index = 0
    # TODO: Fix for GFS wraparound

    pick(ds.header['shape']['y']['min'], ds.header['shape']['y']['step'], ds.header['shape']['y']['count'], lat, "lat", llat)
    pick(ds.header['shape']['x']['min'], ds.header['shape']['x']['step'], ds.header['shape']['x']['count'], lng, "lng", llng)

    cdef long i = 0

    for a in lhour:
        for b in llat:
            for c in llng:
                p = a.lerp * b.lerp * c.lerp
                out[i] = Lerp3(a.index, b.index, c.index, p)
                i += 1

    return 0

cdef double interp3(dataset ds, Lerp3[8] lerps, long variable, long level):
    cdef double r, v

    r = 0
    for i in range(8):
        lerp = lerps[i]
        v = ds[lerp.hour, level, variable, lerp.lat, lerp.lng]
        #print(f"{lerp.hour},{level},{variable},{lerp.lat},{lerp.lng} = {v}")
        r += v * lerp.lerp

    return r

# Searches for the largest index lower than target, excluding the topmost level.
cdef long search(dataset ds, Lerp3[8] lerps, double target):
    cdef long lower, upper, mid
    cdef double test
    
    # GFS
    #lower, upper = 0, 47
    # ECMWF
    #lower, upper = 0, 11
    # MEPS
    lower, upper = 0, 63
    # Header
    lower, upper = 0, (ds.header['shape']['levels'] - 2)

    while lower < upper:
        mid = (lower + upper + 1) / 2
        test = interp3(ds, lerps, VAR_A, mid)
        print(f"search for {target}: {lower} / {upper} = {test}")
        if target <= test:
            upper = mid - 1
        else:
            lower = mid

    return lower

cdef double interp4(dataset ds, Lerp3[8] lerps, Lerp1 alt_lerp, long variable):
    lower = interp3(ds, lerps, variable, alt_lerp.index)
    # and we can infer what the other lerp1 is...
    upper = interp3(ds, lerps, variable, alt_lerp.index + 1)
    return lower * alt_lerp.lerp + upper * (1 - alt_lerp.lerp)
