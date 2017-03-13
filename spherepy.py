"""
Spherical Python Module

Scripts useful when working on a sphere (such as the Earth). 

Shawn Murdzek
sfm5282@psu.edu
"""


# Import Needed Modules

import numpy as np

# Finding the distance between two (latitude, longitude) points

def haversin(A):
    """
    Function that returns the haversine of angle A. The haversine is computed
    using the equation haversin(A) = (1 - cos(A)) / 2
    Inputs:
        A = Angle (radians)
    Outputs:
        hsin = Haversine of angle A
    """    

    hsin = (1 - np.cos(A)) / 2.0

    return hsin


def latlon_dist(lat1, lon1, lat2, lon2, R=6371.0):
    """
    Function that returns the distance between two points on the Earth, taking
    the curvature of the Earth. The formula used here is explained in this pdf:
    https://www.math.ksu.edu/~dbski/writings/haversine.pdf. 
    Inputs:
        lat1 = Latitude of the first point (deg N)
        lon1 = Longitude of the second point (deg E)
        lat2 = Latitude of the second point (deg N)
        lon2 = Longitude of the second point (deg E)
    Outputs:
        dist = Distance between the two points (km)
    Keywords:
        R = Radius of the Earth (km)
    Local Variables:
        hsin = Haversin of alpha, which is explained in the above pdf
    """

    # Convert degrees to radians

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute the haversin(alpha) and the distance

    hsin = (haversin(lat1 - lat2) + np.cos(lat1) * np.cos(lat2) *
            haversin(lon1 - lon2))

    dist = 2.0 * R * np.arcsin(np.sqrt(hsin))

    return dist
    

# Changing between latitude and longitude and Cartesian coordinates

def latlon_to_cart_h(lat, lon, R=6371.0):
    """
    Function that takes a latitude and longitude coordinate and transforms it
    into a Cartesian coordinate. The latitude and longitude that correspond to
    the origin are lat = 0 and lon = 0. This is similar to the transformation
    from spherical to Cartesian coordinates, but the Cartesian grid is not flat
    (it lies on the sphere). The "h" denotes the fact that this Cartesian grid
    lies on the surface of the sphere (like our reference frame on Earth)
    Inputs:
        lat = Latitude of the point (deg N)
        lon = Longitude of the point (deg E)
    Outputs:
        x = X coordinate of the given point (km)
        y = Y coordinate of the given point (km)
    Keywords:
        R = Radius of the sphere (km). The default value is the radius of the
            Earth.
    """

    # Make longitude negative if it is in between 180 and 360 degrees

    if (lon > 180) and (lon < 360):

        lon = 360.0 - lon

    # Transform longitude to a x coordinate

    x = R * np.cos(np.radians(lat)) * np.radians(lon)

    # Transform latitude to a y coordinate

    y = R * np.radians(lat)
    
    return x, y


def latlon_to_cart(lat, lon, R=6371.0):
    """
    Function that transforms a latitude and longitude coordinate into a three
    dimensional Cartesian coordinate using the well-known transformation from
    spherical to Cartesian coordinates. For this script, the origin of the grid
    is the center of the sphere.
    Inputs:
        lat = Latitude of the point (deg N)
        lon = Longitude of the point (deg E)
    Outputs:
        x = X coordinate of the given point (km)
        y = Y coordinate of the given point (km)
        z = Z coordinate of the given point (km)
    Keywords:
        R = Radius of the sphere (km). The default value is the radius of the
            Earth.
    """

    x = R * np.cos(np.radians(lon)) * np.sin(np.radians(lat))
    y = R * np.sin(np.radians(lon)) * np.sin(np.radians(lat))
    z = R * np.cos(np.radians(lat))

    return x, y, z


# Function to find a circle on the surface of a sphere

def sphere_circle(r, lat_0, lon_0, R=6371.0, pts=1000):
    """
    Function that returns an array of points on a circle that lies on the face
    of a sphere using the ideas presented in this math forum:
    http://mathforum.org/library/drmath/view/51882.html. Essentially, the
    equation D = R(alpha) where D = distance and alpha is the angle between the
    position vector for the center of the circle and the position vector for a
    point on the circle. We can then solve for alpha knowing that the dot
    product of these two vcetors is (R ** 2) * (cos(alpha)).
    Inputs:
        r = Radius of the circle, accounting for the curvature of the sphere
            (km).
        lat_0 = Latitude of the center of the sphere (deg N).
        lon_0 = Longitude of the center of the sphere (deg E).
    Outputs:
        points = 2D array of latitude, longitude pairs for points on the circle
            (lat is in deg N, lon is in deg E). Note that each latitude value
            has two longitude values, one in the second column and one in the
            third column.
    Keywords:
        R = Radius of the sphere. The default value is the radius of the Earth
            (km).
        pts = Number of latitude values for which longitude values are
            calculated.
    Local Variables:
        step = Step between consecutive latitude values along the circle
            (radians).
        delta_lon = Distance from lon_0 each longitude point is for each
            respective latitude value (radians).
    """

    # Convert to radians for calculations.

    lat_0 = np.radians(lat_0)
    lon_0 = np.radians(lon_0)

    # Convert inputs to floats.

    r = float(r)
    lat_0 = float(lat_0)
    lon_0 = float(lon_0)
    R = float(R)

    # Define range of latitude values over which arccos() function is defined.
    # This happens when the latitude is between lat_0 - (r /R) and
    # lat_0 + (r/R).

    points = np.zeros([2 * pts, 2])

    step = (2.0 * (r / R)) / float(pts - 1)
    points[:pts, 0] = np.arange(-(r / R), (r / R) + step, step) + lat_0
    points[pts:, 0] = np.arange((r / R), -(r / R) - step, -1.0 * step) + lat_0

    # Calculate the corresponding longitude values for each latitude value along
    # the circle.

    delta_lon = np.arccos((np.cos(r / R) - np.cos(points[:pts, 0]) *
                              np.cos(lat_0)) / (np.sin(points[:pts, 0]) *
                                                np.sin(lat_0)))

    points[:pts, 1] = lon_0 + delta_lon
    points[pts:, 1] = lon_0 - delta_lon

    # Convert back to degrees

    points = np.degrees(points)

    return points


def circle_center(lat_a, lon_a, lat_b, lon_b, r, R=6371.0):
    """
    Function that finds the center of a circle on the surface of a sphere given
    two points along that circle and the radius of the circle. This is done by
    constructing a circle around each of the given points with radius equal to
    the given radius and finding where the two circles intersect. The circles
    will intersect at two different points, giving two possible center points,
    both of which are returned.
    Inputs:
        lat_a = Latitude of point a (deg N).
        lon_a = Longitude of point a (deg E).
        lat_b = Latitude of point b (deg N).
        lon_b = Longitude of point b (deg E).
        r = Radius of the circle on the surface of the sphere (km).
    Outputs:
        center = List of latitude, longitude coordinates that can be centers of
            the circle (deg).
    Keywords:
        R = Radius of the sphere. The default value is the radius of the Earth
            (km).
    Local Variables:
        cir_a = 2D array of points lying along the first circle with center
            lat_a, lon_a (deg).
        cir_b = 2D array of points lying along the second cirle with center
            lat_b, lon_b (deg).
        min_diff = 1D array of the minimum difference for each diff array
            created in the for-loop.
        diff = 1D array of the differences between cir_b and a point on cir_a.
            Note that these are not true distances, they are pseudo-distances.
            
    """

    # Create center list

    center = []

    # Calculate points along the circle centered on point a and point b

    cir_a = sphere_circle(r, lat_a, lon_a)
    cir_b = sphere_circle(r, lat_b, lon_b)

    # Find where the two circles are approximately equal

    min_diff = np.zeros(np.shape(cir_a)[0])

    for i in xrange(np.shape(cir_a)[0]):

        diff = np.zeros(np.shape(cir_a)[0])

        # Find the difference between cir_b and a single point on cir_a using
        # the distance formula to give a pseudo-distance.
        
        diff = np.sqrt((cir_b[:, 0] - cir_a[i, 0]) ** 2.0 +
                       (cir_b[:, 1] - cir_a[i, 1]) ** 2.0)

        min_diff[i] = diff[np.argmin(diff)]

    # Find location along cir_a corresponding to the minimum difference, which
    # is the first circle center.

    center.append(cir_a[np.argmin(min_diff), :])

    # Set the difference for the point above to 90000 so we can find the second
    # smallest difference, which should be the second center.

    min_diff[np.argmin(min_diff)] = 900000

    center.append(cir_a[np.argmin(min_diff), :])

    return center


"""
End spherepy.py
"""
