"""
generate_dataset.py
"""
# This script generates training images for the particle counting model.
import os
import numpy as np
import imageio
from tqdm import tqdm
from scipy.special import erf
   
def normal_gaussian_integrated_within_each_pixel(i, x, sigma):
    """ Compute the integral of the 1D Gaussian.
    Args:
        i (numpy array of ints or int): Pixel index. (e.g., [0, 1, 2, 3, 4, ..., 99])
        x (float): Center position of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
    Returns:
        float: Integral of the Gaussian from i-0.5 to i+0.5.
    """
    norm = 1/2/sigma**2
    # Below is the same as integral(from i-0.5 to i+0.5) [1/2sqrt(pi)*exp(-normalization_factor*(t-x)**2) dt]
    return 0.5*(erf((i-x+0.5)*np.sqrt(norm))-erf((i-x-0.5)*np.sqrt(norm)))

def psfconvolution(peak_info, image_width=512):
    """ Returns the pixel values to be added to the image based on psf convolution.
    Args:
        peak_info (dict): Dictionary containing the following keys:
            x (float): x-coordinate of the peak
            y (float): y-coordinate of the peak
            psf_sigma (float): standard deviation of the PSF
            prefactor (float or list): prefactor of the peak
        image_width (int): width of the image
    Returns:
        numpy.array(dtype=float): image_width x image_width array of the pixel values to be added to the image
    """

    psf_factors_for_pixels_in_x = normal_gaussian_integrated_within_each_pixel(np.arange(image_width), peak_info['x'], peak_info['psf_sigma'])
    psf_factors_for_pixels_in_y = normal_gaussian_integrated_within_each_pixel(np.arange(image_width), peak_info['y'], peak_info['psf_sigma'])

    if isinstance(peak_info['prefactor'], (int, float)): # Case grayscale image
        output = np.outer(psf_factors_for_pixels_in_y, psf_factors_for_pixels_in_x) * peak_info['prefactor']
    else: # Case RGB image
        output = np.array([np.outer(psf_factors_for_pixels_in_y, psf_factors_for_pixels_in_x) * peak_info['prefactor'][i] for i in range(3)])

    return output

def generate_training_images(num_image_per_particle_count, config_file=None, minimum_intensity=1.0, rand_seed=42):

    """ Generate training images for the particle counting model. The images have 0 to 4 particles.
    Parameters:
        n_total_image_count (int): The total number of images to be generated.
        rand_seed (int): The random seed for generating the images.
        
    Returns:
        str: The path of the folder containing the images
    """

    if config_file is not None:
        pass # Write later
    else:
        # First test with random parameters
        # Define the parameters that will set the range of properties of the generated images
        generation_parameters = {
            "min_bg": 10,
            "max_bg": 1000,
            "min_psf_sigma": 2.0,
            "max_psf_sigma": 2.0,
            "max_particles_in_image": 4,
            "min_particles_in_image": 0,

            "image_side_length": 100,
        }

        file_extension = 'tiff'
        # particle_intensity = np.random.randint(min_particle_intensity, max_particle_intensity + 1).astype(float)
        # bg = np.random.randint(min_bg, max_bg + 1).astype(float)
        # psf_sigma = np.random.uniform(min_psf_sigma, max_psf_sigma).astype(float)
        # num_particles = np.random.randint(min_particles_in_image, max_particles_in_image + 1)

    # Set random seed
    np.random.seed(rand_seed)

    # Create the folder to store the images
    trainsetname = "trainset00"
    existing_folders = [f for f in os.listdir() if f.startswith("trainset") and f[8:].isdigit()]
    if existing_folders:
        max_index = max(int(f[8:]) for f in existing_folders)
        trainsetname = f"trainset{max_index + 1:02d}"

    os.makedirs(trainsetname, exist_ok=True)

    bgmin = generation_parameters["min_bg"]
    bgmax = generation_parameters["max_bg"]
    psmin = generation_parameters["min_psf_sigma"]
    psmax = generation_parameters["max_psf_sigma"]
    nummin = generation_parameters["min_particles_in_image"]
    nummax = generation_parameters["max_particles_in_image"]
    image_side_length = generation_parameters["image_side_length"]

    # subfoldername = f"intensity{intmin}-{intmax}-bg{bgmin}-{bgmax}-psf{psmin}-{psmax}-num{nummin}-{nummax}"

    for i in range(nummin, nummax + 1):
        os.makedirs(os.path.join(trainsetname, str(i)), exist_ok=True)

    # Determine the color mode of the image (gray or rgb)
    color_mode = 'gray'

    # Calculate the the total number of images to be generated
    num_total_images = num_image_per_particle_count * (nummax - nummin + 1)

    # Generate the images
    with tqdm(total=num_total_images, desc="Generating Images", unit="image") as pbar:
        # The width of the point spread function (psf) is randomly drawn and same for all particles in the image.
        psf_sigma = float(np.random.uniform(psmin, psmax))
        for n_particles in range(nummin, nummax+1):
            for img_idx in range(num_image_per_particle_count):
                # Initialize the image 
                bg = float(np.random.randint(bgmin, bgmax + 1))
                if color_mode == 'gray':
                    image = np.ones((image_side_length, image_side_length), dtype=float) * bg
                else:   
                    image = [np.ones((image_side_length, image_side_length), dtype=float) * bg[i] for i in range(3)]

                particle_intensity_mean = bg * np.random.uniform(0.1, 100)
                particle_intensity_sd = 0
                # Set the parameters for the particles
                for _ in range(n_particles):
                    # Randomly draw the position of the particle, avoiding the edges of the image
                    x = np.random.uniform(psf_sigma * 2 - 0.5, image_side_length - psf_sigma * 2 - 0.5)
                    y = np.random.uniform(psf_sigma * 2 - 0.5, image_side_length - psf_sigma * 2 - 0.5)

                    # Randomly draw the relative intensity of the particle (mean: 1, std: amp_sd)
                    if color_mode == 'gray':
                        # particle_intensity = np.random.normal(particle_intensity_mean, np.sqrt(particle_intensity_mean))
                        particle_intensity = np.random.normal(particle_intensity_mean, particle_intensity_sd)
                        if particle_intensity < 0:
                            particle_intensity = minimum_intensity

                        # Create peak info dictionary
                        peak_info = {'x': x, 'y': y, 'prefactor': particle_intensity, 'psf_sigma': psf_sigma}


                    else: # Case : rgb
                        pass
                        # particle_intensities = np.array([np.random.normal(particle_intensity_mean, particle_intensity_sd) for i in range(3)])
                        # if np.any(particle_intensities < 0):
                        #     raise ValueError("Randomly drawn particle intensity (at least one of r, g, or b) is less than 0, which is not allowed.")

                        # # Create peak info dictionary
                        # peak_info = {'x': x, 'y': y, 'prefactor': particle_intensities, 'psf_sigma': psf_sigma}

                    # Add the point spread function of the particle to the image
                    image += psfconvolution(peak_info, image_side_length)

                # Add Poisson noise
                image = np.random.poisson(image) # keeping it as float disallows overflowing.
                image_folder_path = os.path.join(trainsetname, str(n_particles))
                img_filename = f"count{n_particles}-index{img_idx}.{file_extension}"
                if file_extension == 'png' and np.any(image > 65535):
                    print(f"Warning: The pixel value(s) of {img_filename} exceeds 65535. Since png can store max 16-bits, such values will be clipped. This mimics saturation in the camera.")
                    image = np.clip(image, 0, 65535)

                # Adjust the shape of the image to match that of png or tiff
                if image.ndim == 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))

                # Save the image
                img_filepath = os.path.join(image_folder_path, img_filename)
                if file_extension == 'png':
                    imageio.imwrite(img_filepath, image.astype(np.uint16))
                elif file_extension == 'tiff':
                    imageio.imwrite(img_filepath, image.astype(np.float32))
                    # imageio.imwrite(img_filepath, image.astype(np.uint32)) # For some reason this renders analysis very slow.

                # Update the progress bar
                pbar.update(1)


generate_training_images(num_image_per_particle_count=10000, config_file=None, rand_seed=42)