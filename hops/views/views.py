from astropy.io import fits
from astropy.time import Time
import ephem
import matplotlib.pyplot as plt
import numpy as np
import os
import pylightcurve
import scipy
import time

from django.conf import settings
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render

from ..models import Dataset,Frame,PhotometrySource,Target


def index(request):
    datasets = Dataset.objects.order_by('name')
    template = loader.get_template('hops/hops.html')
    context = {
        'datasets': datasets,
    }
    return HttpResponse(template.render(context, request))


def sourceSelection(request):
    if request.method == 'POST':
        user_selection = request.POST.get('dataset')
        selected_dataset = Dataset.objects.get(id=user_selection)
    
    if selected_dataset:
        frame = Frame.objects.filter(observation_object__object_name=selected_dataset.name).first()
        fits_img_path = frame.observation_image.url
        data = fits.getdata(fits_img_path)
        image_path = os.path.join(settings.BASE_DIR, 'static/{}.jpg'.format(fits_img_path.strip('/media/.fit')))
        height = float(np.shape(data)[0])
        width = float(np.shape(data)[1])
        fig = plt.figure()
        fig.set_size_inches(width/height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(data, cmap='CMRmap')
        plt.savefig(image_path, dpi = height) 
        plt.close()

        selected_dataset.image_height = height
        selected_dataset.image_width = width
        selected_dataset.save()
    else:
        raise ValueError('A very bad thing happened.')

    template = loader.get_template('hops/source-selection.html')
    context = {
        'selected_dataset': selected_dataset,
        'image_path': '/static/{}.jpg'.format(fits_img_path.strip('/media/.fit'))
    }
    return HttpResponse(template.render(context, request))


def createPhotometrySource(request):

    if request.method == 'POST':
        source_pos_x = float(request.POST.get('sourcePosX'))
        source_pos_y = float(request.POST.get('sourcePosY'))

        source_pos_r, source_pos_u = cartesian_to_polar(source_pos_x, source_pos_x, 0, 0)

        data = PhotometrySource(
            observation_target = Target.objects.get(object_name='Qatar-1b'),
            source_pos_x = source_pos_x,
            source_pos_y = source_pos_y,
            source_pos_r = source_pos_r,
            source_pos_u = source_pos_u,
            source_type = request.POST.get('sourceType')
        )
        data.save()
    return HttpResponse(status=204)


def performPhotometry(request):

    # get variables

    comparisons = PhotometrySource.objects.filter(source_type='comparison')

    # reduction_directory = read_log('pipeline', 'reduction_directory')
    # light_curve_aperture_file = read_log('pipeline', 'light_curve_aperture_file')
    # photometry_directory = read_log('pipeline', 'photometry_directory')
    # photometry_file = read_log('pipeline', 'photometry_file')
    # light_curve_gauss_file = read_log('pipeline', 'light_curve_gauss_file')
    # results_figure = read_log('pipeline', 'results_figure')
    # fov_figure = read_log('pipeline', 'fov_figure')
    # mean_key = read_log('pipeline_keywords', 'mean_key')
    # std_key = read_log('pipeline_keywords', 'std_key')
    # align_x0_key = read_log('pipeline_keywords', 'align_x0_key')
    # align_y0_key = read_log('pipeline_keywords', 'align_y0_key')
    # align_u0_key = read_log('pipeline_keywords', 'align_u0_key')
    # exposure_time_key = read_log('pipeline_keywords', 'exposure_time_key')
    # observation_date_key = read_log('pipeline_keywords', 'observation_date_key')
    # observation_time_key = read_log('pipeline_keywords', 'observation_time_key')
    # star_std = read_log('alignment', 'star_std')
    star_std = 1
    # search_window_std = read_log('alignment', 'search_window_std')
    search_window_std = 10
    # target_ra_dec = read_log('photometry', 'target_ra_dec')
    # sky_inner_aperture = read_log('photometry', 'sky_inner_aperture')
    # sky_outer_aperture = read_log('photometry', 'sky_outer_aperture')
    sky_inner_aperture = 1.7
    sky_outer_aperture = 2.4
    max_comparisons = comparisons.count()
    targets_r_position = list(entry['source_pos_r'] for entry in comparisons.values())
    targets_u_position = list(entry['source_pos_u'] for entry in comparisons.values())
    targets_aperture = list(entry['aperture_size'] for entry in comparisons.values())

    # science = glob.glob('{0}{1}*.f*t*'.format(reduction_directory, os.sep))
    # science.sort()
    reduced_aligned_files = Frame.objects.all().order_by('observation_date', 'observation_start_time')
    photometric_target = PhotometrySource.objects.filter(source_type='target')
    photometric_comparison = PhotometrySource.objects.filter(source_type='comparison')

    targets_files = []
    targets_hjd = []
    targets_x_position = []
    targets_y_position = []
    targets_x_std = []
    targets_y_std = []
    targets_gauss_flux = []
    targets_gauss_sky = []
    targets_aperture_flux = []
    targets_aperture_sky = []

    # for each science_file
    percent = 0
    lt0 = time.time()
    for counter, file in enumerate(reduced_aligned_files):

        local_time = file.observation_start_time
        julian_date = Time(local_time).jd

        target_ra = file.observation_object.object_RA
        target_dec = file.observation_object.object_DEC

        heliocentric_julian_date = jd_to_hjd(target_ra, target_dec, julian_date)
        targets_hjd.append(heliocentric_julian_date)

        # calculate gauss position, flux and sky and aperture flux and sky
        ref_x_position = file.align_x0
        ref_y_position = file.align_y0
        ref_u_position = file.align_y0

        data = fits.open(file.observation_image.url)[1].data
        header = fits.open(file.observation_image.url)[1].header

        for target in range(max_comparisons):
            if file.aperture_size > 0:
                norm, floor, x_mean, y_mean, x_std, y_std = \
                    fit_2d_gauss(data,
                                 predicted_x_mean=(ref_x_position + targets_r_position[target] *
                                                   np.cos(ref_u_position + targets_u_position[target])),
                                 predicted_y_mean=(ref_y_position + targets_r_position[target] *
                                                   np.sin(ref_u_position + targets_u_position[target])),
                                 search_window=search_window_std * star_std)

                targets_x_position.append(x_mean)
                targets_y_position.append(y_mean)
                targets_x_std.append(x_std)
                targets_y_std.append(y_std)
                targets_gauss_flux.append(2 * np.pi * norm * x_std * y_std)
                targets_gauss_sky.append(floor)

                flux_area = data[int(y_mean) - targets_aperture[target]:
                                int(y_mean) + targets_aperture[target] + 1,
                                int(x_mean) - targets_aperture[target]:
                                int(x_mean) + targets_aperture[target] + 1]
                flux_pixels = (2 * targets_aperture[target] + 1) ** 2
                flux = np.sum(flux_area)

                sky_area_1 = int(sky_inner_aperture * targets_aperture[target])
                sky_area_2 = int(sky_outer_aperture * targets_aperture[target])
                data[int(y_mean) - sky_area_1:int(y_mean) + sky_area_1 + 1,
                             int(x_mean) - sky_area_1:int(x_mean) + sky_area_1 + 1] = 0
                sky_area = data[int(y_mean) - sky_area_2:int(y_mean) + sky_area_2 + 1,
                                        int(x_mean) - sky_area_2:int(x_mean) + sky_area_2 + 1]
                sky_area = sky_area[np.where((sky_area > 0) &
                                             (sky_area < header['MEAN'] + 3 * header['STD']))]
                sky = np.sum(sky_area)
                sky_pixels = sky_area.size

                targets_aperture_flux.append(flux - flux_pixels * sky / sky_pixels)
                targets_aperture_sky.append(sky / sky_pixels)

#     # save results, create photometry directory and move results there
    measurements_number = counter + 1
    targets_number = len(targets_x_position) // measurements_number
    comparisons_number = len(targets_x_position) // measurements_number - 1

    targets_hjd = np.array(targets_hjd)
    targets_x_position = np.swapaxes(np.reshape(targets_x_position, (measurements_number, targets_number)), 0, 1)
    targets_y_position = np.swapaxes(np.reshape(targets_y_position, (measurements_number, targets_number)), 0, 1)
    targets_x_std = np.swapaxes(np.reshape(targets_x_std, (measurements_number, targets_number)), 0, 1)
    targets_y_std = np.swapaxes(np.reshape(targets_y_std, (measurements_number, targets_number)), 0, 1)
    targets_gauss_flux = np.swapaxes(np.reshape(targets_gauss_flux, (measurements_number, targets_number)), 0, 1)
    targets_gauss_sky = np.swapaxes(np.reshape(targets_gauss_sky, (measurements_number, targets_number)), 0, 1)
    targets_aperture_flux = np.swapaxes(np.reshape(targets_aperture_flux,
                                                   (measurements_number, targets_number)), 0, 1)
    targets_aperture_sky = np.swapaxes(np.reshape(targets_aperture_sky,
                                                  (measurements_number, targets_number)), 0, 1)

    targets_results = [targets_hjd] + (list(targets_x_position) + list(targets_y_position) + list(targets_x_std) +
                                       list(targets_y_std) + list(targets_gauss_flux) + list(targets_gauss_sky) +
                                       list(targets_aperture_flux) + list(targets_aperture_sky))

#     np.savetxt(photometry_file,
#                np.swapaxes(targets_results, 0, 1))

#     np.savetxt(light_curve_gauss_file,
#                np.swapaxes([targets_hjd, targets_gauss_flux[0] / np.sum(targets_gauss_flux[1:], 0)], 0, 1))

#     np.savetxt(light_curve_aperture_file,
#                np.swapaxes([targets_hjd, targets_aperture_flux[0] / np.sum(targets_aperture_flux[1:], 0)], 0, 1))

#     if not os.path.isdir(photometry_directory):
#         os.mkdir(photometry_directory)
#     else:
#         fi = 2
#         while os.path.isdir('{0}_{1}'.format(photometry_directory, str(fi))):
#             fi += 1
#         photometry_directory = '{0}_{1}'.format(photometry_directory, str(fi))
#         os.mkdir(photometry_directory)

#     root = Tk()

    if comparisons_number > 1:
        f = plt.figure()
        f.set_figwidth(7)
        # f.set_figheight(0.8 * root.winfo_screenheight() / f.get_dpi())
        ax = f.add_subplot(comparisons_number + 1, 1, 1)
    else:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)

    ax.plot(targets_hjd - targets_hjd[0], targets_aperture_flux[0] / np.sum(targets_aperture_flux[1:], 0)
            / np.median(targets_aperture_flux[0] / np.sum(targets_aperture_flux[1:], 0)), 'ko', ms=3)
    ax.plot(targets_hjd - targets_hjd[0], targets_gauss_flux[0] / np.sum(targets_gauss_flux[1:], 0)
            / np.median(targets_gauss_flux[0] / np.sum(targets_gauss_flux[1:], 0)), 'ro', ms=3, mec='r')
    ax.tick_params(labelbottom='off')
    ax.set_title(r'$\mathrm{Target}$')

    if comparisons_number > 1:
        for comp in range(comparisons_number):
            test_aperture_flux = list(targets_aperture_flux[1:])
            test_gauss_flux = list(targets_gauss_flux[1:])
            del test_aperture_flux[comp]
            del test_gauss_flux[comp]
            ax = f.add_subplot(comparisons_number + 1, 1, comp + 2)
            ax.plot(targets_hjd - targets_hjd[0], targets_aperture_flux[1:][comp] / np.sum(test_aperture_flux, 0)
                    / np.median(targets_aperture_flux[1:][comp] / np.sum(test_aperture_flux, 0)), 'ko', ms=3)
            ax.plot(targets_hjd - targets_hjd[0], targets_gauss_flux[1:][comp] / np.sum(test_gauss_flux, 0)
                    / np.median(targets_gauss_flux[1:][comp] / np.sum(test_gauss_flux, 0)),
                    'ro', ms=3, mec='r')
            ax.tick_params(labelbottom='off')
            ax.set_title(r'${0}{1}{2}$'.format('\mathrm{', 'Comparison \, {0}'.format(comp + 1), '}'))

    ax.tick_params(labelbottom='on')
    ax.set_xlabel(r'$\mathrm{\Delta t} \ \mathrm{[days]}$', fontsize=20)
    f.text(0.03, 0.5, r'$\mathrm{relative} \ \mathrm{flux}$', fontsize=20,
           ha='center', va='center', rotation='vertical')
    plt.savefig('static/photometry.png')

    template = loader.get_template('hops/photometry-results.html')
    context = {
        'targets_results': targets_results,
        'image_path': '/static/photometry.png'
    }

    return HttpResponse(template.render(context, request))


def jd_to_hjd(ra_target, dec_target, julian_date, degrees=False):

    if degrees:
        ra_target *= np.pi / 180
        dec_target *= np.pi / 180
    else:
        k = ephem.Equatorial(ra_target, dec_target)
        ra_target = k.ra
        dec_target = k.dec

    sun = ephem.Sun()
    sun.compute(ephem.date(julian_date - 2415020))
    ra_sun, dec_sun = float(sun.ra), float(sun.dec)

    a = 149597870700.0 / ephem.c
    b = np.sin(dec_target) * np.sin(dec_sun)
    c = np.cos(dec_target) * np.cos(dec_sun) * np.cos(ra_target - ra_sun)

    heliocentric_julian_date = julian_date - (a * (b + c)) / (24.0 * 60.0 * 60.0)

    return heliocentric_julian_date


def fit_2d_gauss(data_array, predicted_x_mean=None, predicted_y_mean=None, search_window=None):

    found = False

    if not predicted_x_mean:
        predicted_x_mean = np.where(data_array == np.max(data_array))[1][0]

    if not predicted_y_mean:
        predicted_y_mean = np.where(data_array == np.max(data_array))[0][0]

    if not search_window:
        search_window = max(data_array.shape)

    norm, floor, x_mean, y_mean, x_std, y_std = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    while not found and search_window < max(data_array.shape):

        y_min = int(max(predicted_y_mean - search_window, 0))
        y_max = int(min(predicted_y_mean + search_window, len(data_array)))
        x_min = int(max(predicted_x_mean - search_window, 0))
        x_max = int(min(predicted_x_mean + search_window, len(data_array[0])))

        cropped_data_array = data_array[y_min:y_max, x_min:x_max]
        cropped_x_data_array = np.arange(x_min, x_max)
        cropped_y_data_array = np.arange(y_min, y_max)

        norm = data_array[int(predicted_y_mean)][int(predicted_x_mean)] - np.min(cropped_data_array)
        floor = np.min(cropped_data_array)
        x_mean = cropped_x_data_array[np.where(cropped_data_array == np.max(cropped_data_array))[1][0]]
        y_mean = cropped_y_data_array[np.where(cropped_data_array == np.max(cropped_data_array))[0][0]]
        x_std = max(1.0, np.abs(cropped_x_data_array[np.argmin(
            np.abs(np.sum(cropped_data_array, 0) - np.max(np.sum(cropped_data_array, 0)) / 2))] - x_mean))
        y_std = x_std
        theta = 0.0

        try:
            [norm, floor, x_mean, y_mean, x_std, y_std, theta], covariance = \
                scipy.optimize.curve_fit(function_2d_gauss,
                          np.meshgrid(cropped_x_data_array, cropped_y_data_array),
                          cropped_data_array.flatten(),
                          p0=[norm, floor, x_mean, y_mean, x_std, y_std, theta])

            x_mean += 0.5
            y_mean += 0.5
            x_std = abs(x_std)
            y_std = abs(y_std)

            if norm > 3 * np.sqrt(covariance[0][0]):
                found = True

            else:
                search_window *= 2.0

        except RuntimeError:

            search_window *= 2.0

    if found:
        return norm, floor, x_mean, y_mean, x_std, y_std

    else:
        print('fit_2d_gauss: could not find a 2D Gaussian')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def function_2d_gauss(xy_array, model_norm, model_floor,
                          model_x_mean, model_y_mean, model_x_std, model_y_std, model_theta):

        x_array, y_array = xy_array

        a = (np.cos(model_theta) ** 2) / (2 * model_x_std ** 2) + (np.sin(model_theta) ** 2) / (2 * model_y_std ** 2)
        b = -(np.sin(2 * model_theta)) / (4 * model_x_std ** 2) + (np.sin(2 * model_theta)) / (4 * model_y_std ** 2)
        c = (np.sin(model_theta) ** 2) / (2 * model_x_std ** 2) + (np.cos(model_theta) ** 2) / (2 * model_y_std ** 2)

        return (model_floor + model_norm * np.exp(- (a * ((x_array - model_x_mean) ** 2)
                                                  + 2.0 * b * (x_array - model_x_mean) * (y_array - model_y_mean)
                                                  + c * ((y_array - model_y_mean) ** 2)))).flatten()


def cartesian_to_polar(x_position, y_position, x_ref_position, y_ref_position):

    x_position, y_position = float(x_position), float(y_position)
    x_ref_position, y_ref_position = float(x_ref_position), float(y_ref_position)

    radius = np.sqrt((x_position - x_ref_position) ** 2 + (y_position - y_ref_position) ** 2)

    if (x_position - x_ref_position) > 0:
        if (y_position - y_ref_position) >= 0:
            angle = np.arctan((y_position - y_ref_position) / (x_position - x_ref_position))
        else:
            angle = 2.0 * np.pi + np.arctan((y_position - y_ref_position) / (x_position - x_ref_position))
    elif (x_position - x_ref_position) < 0:
        angle = np.arctan((y_position - y_ref_position) / (x_position - x_ref_position)) + np.pi
    else:
        if (y_position - y_ref_position) >= 0:
            angle = np.pi / 2
        else:
            angle = -np.pi / 2

    return radius, angle