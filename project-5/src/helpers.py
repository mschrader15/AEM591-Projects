import math
from typing import List, Union

import numpy as np

RAD_2_DEGREE = 180 / math.pi


def calculate_dubins(
    x0: tuple = (0, -15, -90),
    x1: tuple = (-5, 20, -180),
    r: int = 5,
    step_size: float = 0.01,
) -> list:
    import dubins

    x0 = (*x0[:-1], x0[-1] / RAD_2_DEGREE)
    x1 = (*x1[:-1], x1[-1] / RAD_2_DEGREE)
    path = dubins.shortest_path(x0, x1, r)
    optimal_path, _ = path.sample_many(step_size)
    return optimal_path


def normalize_radians(x: Union[List, float]) -> Union[List, float]:

    if isinstance(x, list):
        return list(map(normalize_radians, x))
    else:
        x = x % (2 * np.pi)
        if x > np.pi:
            x -= 2 * np.pi
        return x


def confidence_ellipse(x, y, cov, n_std=1.96, size=100):
    """
    Largely from https://gist.github.com/dpfoose/38ca2f5aee2aea175ecc6e599ca6e973

    Get the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    size : int
        Number of points defining the ellipse
    Returns
    -------
    String containing an SVG path for the ellipse

    References (H/T)
    ----------------
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    https://community.plotly.com/t/arc-shape-with-path/7205/5
    """
    import numpy as np

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack(
        [ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)]
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array(
        [
            [np.cos(np.pi / 4), np.sin(np.pi / 4)],
            [-np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    scale_matrix = np.array([[x_scale, 0], [0, y_scale]])
    ellipse_coords = (
        ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix
    )

    path = f"M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}"
    for k in range(1, len(ellipse_coords)):
        path += f"L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}"
    path += " Z"
    return path
