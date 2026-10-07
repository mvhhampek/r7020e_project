import numpy as np
from utils import pixel_to_xyz
from utils import classify_detection

#   X = (u - cx) * Z / fx
#   Y = (v - cy) * Z / fy
#   Z = depth
# P (3x4):
#   [[fx,  0, cx, 0],
#    [ 0, fy, cy, 0],
#    [ 0,  0,  1, 0]]


def test_pixel_to_xyz_at_optical_center():
    u = 100
    v = 50
    depth = 1

    P = np.array([[1, 0, u, 0], [0, 1, v, 0], [0, 0, 1, 0]])
    
    expected_result = np.array([0, 0, depth], dtype=np.float64)

    np.testing.assert_array_almost_equal(pixel_to_xyz(v, u, depth, P), expected_result)



def test_pixel_to_xyz_known_values():
    fx = 400
    fy = 250
    cx = 640
    cy = 320
    P = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])

    v = 100
    u = 50
    depth = 2000
    expected_result = np.array([-2950, -1760, 2000])

    np.testing.assert_array_almost_equal(pixel_to_xyz(v, u, depth, P), expected_result)




def test_depth_nan_values():
    nan_values = np.array([100, 50, 0]) # all values are too low (<200)
    assert classify_detection(nan_values) is None

def test_depth_valid_values():
    valid_values = np.arange(1000,4000) # values spread over 1000-4000 without gaps
    assert classify_detection(valid_values) == True

def test_depth_invalid_values():
    invalid_values = np.repeat(np.arange(1900,2100), 10) # values without alot of spread
    assert classify_detection(invalid_values) == False
