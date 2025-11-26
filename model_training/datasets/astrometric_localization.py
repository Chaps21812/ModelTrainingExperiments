import numpy as np
# import cupy as cp
from photutils.datasets import make_noise_image
from photutils.psf import IntegratedGaussianPRF
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry
from astropy.table import QTable

def detect_stars(stacked_frames):
    '''
    Utilizes a integrated gaussian point spread function to identify stars.

    Notes: Current configuration of the PSF model works for a scale of 4-5 arcmin
           sized image. Will make this more adjustable with calculations if needed.

    Input: The stacked frames to be processed for astrometric localization. Works
           best when background has already been corrected.

    Output: A numpy array of shape (2, N) where N is the number of stars extracted. 
            Contains the x and y pixel coordinates of each extracted star.
    '''
    psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
    fit_shape = (11, 11)
    finder = DAOStarFinder(100.0, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                        aperture_radius=4)
    phot = psfphot(stacked_frames)
    stars = np.asarray([[x, y] for x, y in zip(phot['x_fit'], phot['y_fit'])])
    return stars

def match_to_catalogue(extracted_stars, scales={4,5}):
    '''
    Matches a list of stars to a skyfield, allowing for an astrometric fit.

    Note: Scales 4 to 5 are required for current SatSim configs.

    Input: List (x, y) coordinates of stars.

    Out: Astrometric solution from astrometry.net if successful, None otherwise.
    '''
    solver = astrometry.Solver(
        astrometry.series_5200.index_files(
            cache_directory= 'astrometry_cache/portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE',
            scales = {4, 5},
        )
    )
    solution = solver.solve(
        stars=extracted_stars,
        size_hint=None,
        position_hint=None,
        solution_parameters=astrometry.SolutionParameters(),
    )
    if solution.has_match():
        return solution
    return None

def skycoord_to_pixels(astrometric_solution):
    '''
    Converts the solution's sky coordinates to pixel coordinates.

    Input: Astrometric solution

    Output: List of (x, y) coordinates of stars.
    '''
    wcs = astrometric_solution.best_match().astropy_wcs()
    pixels = wcs.all_world2pix([[star.ra_deg, star.dec_deg] for star in astrometric_solution.best_match().stars], 0,)
    return pixels

def solution_to_image_coords(image_stars, solution_stars):
    '''
    The astrometric solution pixel coordinates do not always match with the provided image due to instrument
    calibration differences. Out of the initial extracted stars, this function returns the list of stars that
    do match with the astrometric solution using a transformation matrix. Model is created using sklearns 
    transform model within astroalign.

    Inputs: The image star list of (x, y) coords and the astrometric solution list of (x, y) coords

    Output: Transformation Matric and matched stars in the form of (x, y) coords
    '''
    transf, (s_list, t_list) = aa.find_transform(image_stars, solution_stars)
    dst_calc = aa.matrix_transform(solution_stars, transf.inverse)
    return (transf, dst_calc)

# def apply_starmask(s_list, stacked_image, images):
#     '''
#     Based on the inputted (x, y) coordinates, creates a mask which eliminates known stars from the image

#     Input: List of (x, y) coordinates of stars

#     Output: A mask that removes stars
#     '''
#     psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
#     fit_shape = (11, 11)
#     finder = DAOStarFinder(100.0, 5)
#     psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
#                         aperture_radius=4)
#     init_params = QTable()
#     init_params['x'] = s_list[:,0] 
#     init_params['y'] = s_list[:,1] 
#     phot = psfphot(cp.ndarray.get(stacked_image), init_params = init_params)

#     processed_images = [psfphot.make_residual_image(cp.ndarray.get(file), (9, 9)).clip(min=0) for file in images]
#     return processed_images