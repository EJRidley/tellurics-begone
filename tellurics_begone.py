import sys
import glob
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt, gridspec as gspec


plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 22,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'mathtext.default': 'regular',
    'axes.prop_cycle': plt.cycler(
        color=['#583ea3', '#ff7f00', '#4daf4a', '#f781bf', '#164608', '#377eb8', '#999999', '#ff1a1c', '#dede00']
    )  # modified from https://gist.github.com/thriveth/8560036
})


def norm_gaussian(x, mu, sigma):
    return np.exp((-.5)*(((x-mu)/sigma)**2)) / (sigma * np.sqrt(2*np.pi))


def gaussian_convolve(x, y, sigma, result_axis=None, fill_value=1., mask_thresh=3):
    """
    Convolve a 2-d set of data with a gaussian kernel.

    :param x: x axis of the data
    :param y: y axis of the data
    :param sigma: standard deviation of the gaussian
    :param result_axis: optional, defaults to None, new x axis to map the convolved y values onto
    :param fill_value: optional, defaults to 1., default value if there are no y values close to result axis
    :param mask_thresh: optional, defaults to 3, only consider y values within mask_thresh*sigma of each point
    :return: convolved y values
    """

    # if a result axis is not specified, use the x axis of the data
    if result_axis is None:
        result_axis = x
    # create a default axis with the fill value
    conv = np.ones(len(result_axis)) * fill_value

    # define masks to limit the size of each kernel
    masks = np.array([np.logical_and(mu - mask_thresh*sigma < x, x < mu + mask_thresh*sigma) for mu in result_axis])
    # only consider points where the masks aren't empty
    valid = np.array([np.sum(m) for m in masks]) > 0
    # define the kernels as normalised gaussians centred on each point
    kernels = [norm_gaussian(x[mask], mu, sigma) for mu, mask in zip(result_axis[valid], masks[valid])]
    # apply the normalised kernels to the data around each point and sum each result
    conv[valid] = [np.sum((kernel/np.sum(kernel))*y[mask]) for kernel, mask in zip(kernels, masks[valid])]

    return np.array(conv)


def apply_single_correction(params, axis, spectrum, model, resolution, over_smoothing=10, mask=None):
    # unpack the correction parameters
    scale = params[0]
    axis_shift = params[1]

    # bogus mask if not specified
    if mask is None:
        mask = np.ones(len(axis), dtype=bool)

    # convolve the telluric model to the shifted axis
    conv_model = gaussian_convolve(model[:, 0]+axis_shift, model[:, 1], resolution/2,
                                   result_axis=axis[mask], fill_value=1.)

    # calculate the corrected spectrum
    correction = scale*conv_model + (1-scale)
    corrected = spectrum[mask] / correction

    # calculate the convolution metric
    conv_corrected = gaussian_convolve(axis[mask], corrected, over_smoothing*resolution/2)
    metric = np.sum(((corrected - conv_corrected)**2)/(conv_corrected**2))

    return metric, corrected


def remove_tellurics(spectrum, resolution=None, max_shift=None, fit_bounds=None, verbose=True):
    """
    Remove telluric features from an astronomical spectrum.
    Telluric models are expected to be stored in test_spectra/

    :param spectrum: 2-D array-like of shape [:, 2], raw spectrum in two columns for x and y
    :param resolution: optional, defaults to 10, spectral resolution in angstroms
    :param max_shift: optional, defaults to 2*resolution, max wavelength shift for fitting
    :param fit_bounds: optional, 2-length array-like, defaults to None, lower and upper wavelength bounds for fitting
    :param verbose: optional, defaults to True, verbosity of output
    :return: corrected intensities
    """
    # set defaults
    if resolution is None:
        if verbose:
            print('\tNo resolution specified, using 10 angstroms...')
        resolution = 10
    else:
        if verbose:
            print(f'\tTelluric models will be convolved to {resolution} angstrom(s).')

    if max_shift is None:
        max_shift = resolution*2

    # unpack spectrum
    wavelengths = spectrum[:, 0]
    intensities = spectrum[:, 1]

    # import telluric models
    tell_model_paths = sorted(glob.glob('telluric_models/*'))

    tell_models = []
    for path in tell_model_paths:
        tell_models.append(np.genfromtxt(path))

    # initialise array for corrected intensities
    corrected_intensities = np.copy(intensities)

    # loop through the telluric models and remove them one at a time
    for i, tell_model in enumerate(tell_models):
        if verbose:
            print(f'\tRemoving {tell_model_paths[i]}...')

        # find the wavelength range of the telluric model and create a mask
        tell_bounds = (np.nanmin(tell_model[:, 0]), np.nanmax(tell_model[:, 0]))
        tell_mask = np.logical_and(
            tell_bounds[0] <= wavelengths, wavelengths <= tell_bounds[1]
        )
        # if a fit range is specified, include them in the telluric mask
        if fit_bounds is not None:
            tell_mask = np.logical_and(
                tell_mask,
                np.logical_and(fit_bounds[0] <= wavelengths, wavelengths <= fit_bounds[1])
            )

        # first pass convolution of the telluric model to determine the maximum possible correction
        test_conv = gaussian_convolve(tell_model[:, 0], tell_model[:, 1], resolution / 2,
                                      result_axis=wavelengths[tell_mask], fill_value=1.)

        min_correction = 1e-2
        max_scale = (min_correction - 1) / (np.min(test_conv) - 1)

        # ignore points where the telluric model has neglidgible value
        test_mask = np.array(abs(test_conv - 1) > 1e-5)
        # ignore NaNs
        nan_mask = np.isnan(intensities[tell_mask])

        # fit the correction, minimising the convolution metric
        res = minimize(
            lambda p: apply_single_correction(
                p, wavelengths[tell_mask][~nan_mask],
                corrected_intensities[tell_mask][~nan_mask],
                tell_model, resolution, mask=test_mask[~nan_mask]
            )[0],
            x0=np.array([1., 0.]), bounds=((0.1, max_scale), (-max_shift, max_shift))
        )

        if verbose:
            print(f'\t\tscale: {res.x[0]}\n\t\taxis shift: {res.x[1]}\n')

        # apply the correction using the optimised parameters
        corrected_intensities = apply_single_correction(
            res.x, wavelengths, corrected_intensities, tell_model, resolution
        )[1]

    return corrected_intensities


def plot_correction(spectrum, corrected, resolution=None):
    if resolution is None:
        resolution = 10

    fig = plt.figure()
    gs = gspec.GridSpec(5, 1, hspace=0)
    ax1 = fig.add_subplot(gs[:-1, 0])
    ax2 = fig.add_subplot(gs[-1, 0])

    ax1.step(spectrum[:, 0], spectrum[:, 1], where='mid', label='raw')
    ax1.step(spectrum[:, 0], corrected, where='mid', label='corrected')

    ax1.legend()
    ax1.set(xticklabels=[], yticks=[])

    # convolve telluric models to resolution used in correction
    tell_model_paths = sorted(glob.glob('telluric_models/*'))

    for path in tell_model_paths:
        tell = np.genfromtxt(path)
        conv_tell = gaussian_convolve(tell[:, 0], tell[:, 1], resolution / 2, fill_value=1.,
                                      result_axis=spectrum[:, 0])

        ax2.step(spectrum[:, 0], conv_tell, where='mid', label=path.split('/')[-1])

    ax2.legend(fontsize='x-small')
    ax2.set(xlabel='Observed Wavelength [$\\AA$]')

    plt.show()
    plt.close(fig)


def testing():
    for path in glob.glob('test_spectra/*'):
        print('Correcting', path, '...')
        spectrum = np.genfromtxt(path)
        corrected = remove_tellurics(spectrum)

        plot_correction(spectrum, corrected)


def main():
    try:
        # parse arguments
        path = sys.argv[1]
        resolution = None if len(sys.argv) < 3 else float(sys.argv[2])

        # import spectrum
        spectrum = np.genfromtxt(path)

        # perform correction
        print('Correcting', path, '...')
        corrected = remove_tellurics(spectrum, resolution=resolution)

        # save result
        new_path = path.split('/')[-1].split('.')[0] + '_tc.txt'
        np.savetxt(new_path, np.array([spectrum[:, 0], corrected]).T)
        print(f'Saved to {new_path}')

        # plot correction
        plot_correction(spectrum, corrected, resolution)
    except IndexError:
        # no arguments
        ans = input('No path specified, would you like to run on test spectra? [n]')
        if ans == 'y':
            testing()


if __name__ == '__main__':
    main()
