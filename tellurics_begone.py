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
    scale = params[0]
    axis_shift = params[1]

    if mask is None:
        mask = np.ones(len(axis), dtype=bool)

    conv_model = gaussian_convolve(model[:, 0]+axis_shift, model[:, 1], resolution/2,
                                   result_axis=axis[mask], fill_value=1.)

    correction = scale*conv_model + (1-scale)
    corrected = spectrum[mask] / correction

    conv_corrected = gaussian_convolve(axis[mask], corrected, over_smoothing*resolution/2)
    metric = np.sum(((corrected - conv_corrected)**2)/(conv_corrected**2))

    return metric, corrected


def remove_tellurics(spectrum, resolution=10, max_shift=None, fit_bounds=None, verbose=True):
    if max_shift is None:
        max_shift = resolution

    wavelengths = spectrum[:, 0]
    intensities = spectrum[:, 1]

    tell_model_paths = sorted(glob.glob('telluric_models/*'))

    tell_models = []
    for path in tell_model_paths:
        tell_models.append(np.genfromtxt(path))

    corrected_intensities = np.copy(intensities)

    for i, tell_model in enumerate(tell_models):
        if verbose:
            print(f'\t{tell_model_paths[i]}')

        tell_bounds = (np.nanmin(tell_model[:, 0]), np.nanmax(tell_model[:, 0]))
        tell_mask = np.logical_and(
            tell_bounds[0] <= wavelengths, wavelengths <= tell_bounds[1]
        )
        if fit_bounds is not None:
            tell_mask = np.logical_and(
                tell_mask,
                np.logical_and(fit_bounds[0] <= wavelengths, wavelengths <= fit_bounds[1])
            )

        test_conv = gaussian_convolve(tell_model[:, 0], tell_model[:, 1], resolution / 2,
                                      result_axis=wavelengths[tell_mask], fill_value=1.)

        min_correction = 1e-2
        max_scale = (min_correction - 1) / (np.min(test_conv) - 1)

        test_mask = np.array(abs(test_conv - 1) > 1e-5)
        nan_mask = np.isnan(intensities[tell_mask])

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

        corrected_intensities = apply_single_correction(
            res.x, wavelengths, corrected_intensities, tell_model, resolution
        )[1]

    return corrected_intensities


def main():
    for path in glob.glob('test_spectra/*'):
        print(path)
        spectrum = np.genfromtxt(path)
        corrected = remove_tellurics(spectrum)

        fig = plt.figure()
        gs = gspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        ax.step(spectrum[:, 0], spectrum[:, 1], where='mid', label='raw')
        ax.step(spectrum[:, 0], corrected, where='mid', label='corrected')

        ax.set(xlabel='Observed Wavelength [$\\AA$]')
        ax.legend()

        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
