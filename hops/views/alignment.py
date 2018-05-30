from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astropy.io import fits as pf

from django.shortcuts import render

from ..models import *

###HAVE A LOOK IN HOPS.BASICS TO SEE IF IM MISSING ANYTHING!!!


class aligndetails():
    '''
	    THESE ARE ALL THE VARIABLES WE NEED HERE, previous code compared to what we're going to call it in the code. here we read in all the variables from the model? evnetuall it will look like: MEAN = frame.objects(bla bla bla), STD = frame.obj(bla)     etc..    

    WHAT THE VAR IS IN THE CODE                              WHAT WE'LL CALL IT IN MODEL..
    mean_key = read_log('pipeline_keywords', 'mean_key')    #MEAN
    std_key = read_log('pipeline_keywords', 'std_key')        #STD
    align_star_area_key = read_log('pipeline_keywords', 'align_star_area_key')  #ALIGN_SA
    align_x0_key = read_log('pipeline_keywords', 'align_x0_key')                 #ALIGN_X0
    align_y0_key = read_log('pipeline_keywords', 'align_y0_key')		 #ALIGN_Y0
    align_u0_key = read_log('pipeline_keywords', 'align_u0_key')                 #ALIGN_U0
    frame_low_std = read_log('windows', 'frame_low_std')                         #frame_low_std                              
    frame_upper_std = read_log('windows', 'frame_upper_std')		
    burn_limit = read_log('alignment', 'burn_limit')
    star_std = read_log('alignment', 'star_std')
    search_window_std = read_log('alignment', 'search_window_std')
    shift_tolerance = read_log('alignment', 'shift_tolerance_p')
    '''	

class functs_align():
	'''	
	defs used in align method
	########## CHECK IF WAY IVE USED SELF IS LEGIT!!!!!!!!!!! ##############
	'''
	def find_centroids(data_array, x_low=0, x_upper=None, y_low=0, y_upper=None,
                   x_centre=None, y_centre=None,
                   mean=None, std=None, std_limit=3.0, burn_limit=None, star_std=2,
                   flux_order=False):

    		if not x_upper:
        		x_upper = data_array.shape[1]
    		else:
        		x_upper = min(x_upper, len(data_array[0]))

    		if not y_upper:
        		y_upper = data_array.shape[0]
    		else:
        		y_upper = min(y_upper, len(data_array))

    		x_low = max(0, x_low)
    		y_low = max(0, y_low)

    		if not x_centre:
        		x_centre = data_array.shape[1] / 2

    		if not y_centre:
       			y_centre = data_array.shape[0] / 2

    		if not mean or not std:

        		fit_norm, fit_floor, fit_mean, fit_std = self.fit_distribution1d_gaussian(data_array)      #do i need an init self to do this?

        		if np.isnan(fit_norm):
            			fit_mean = np.mean(data_array)
            			fit_std = np.std(data_array)

        		if not mean:
            			mean = fit_mean

        		if not std:
            			std = fit_std

    		if not burn_limit:
        		burn_limit = np.max(data_array)

    		data_array = np.full_like(data_array[y_low:y_upper, x_low:x_upper], data_array[y_low:y_upper, x_low:x_upper])

    		noise_limit = mean + std_limit * std

    		test = []

    		for i in range(-star_std, star_std + 1):
        		for j in range(-star_std, star_std + 1):
            			test.append(np.roll(np.roll(data_array, i, 0), j, 1))

    		stars = np.where((data_array < burn_limit) & (np.max(test, 0) == data_array) & (np.median(test, 0) > noise_limit))
    		stars = [np.sqrt((stars[1] + x_low - x_centre) ** 2 + (stars[0] + y_low - y_centre) ** 2),
             	stars[1] + x_low, stars[0] + y_low, np.sum(test, 0)[stars]]
    		stars = np.swapaxes(stars, 0, 1)

    		del data_array
    		del test

    		if not flux_order:
        		stars = stars[stars[:, 0].argsort()]
        		return stars

    		else:
        		stars = stars[stars[:, 3].argsort()][::-1]
        		return stars


		#######################################################################



	def fit_distribution1d_gaussian(data_array, step=None, binning=None):

    			bins_i, bins_f, counts = self.distribution1d(data_array, step=step, binning=binning)

    			norm, floor, mean, std = self.fit_gaussian(0.5 * (bins_i + bins_f), counts)

    			return norm, floor, mean, std


		########################################################################


	def fit_gaussian(data_x_array, data_y_array):

    			mean = data_x_array[np.argmax(data_y_array)]
    			std = np.abs(data_x_array[np.argmin(np.abs(data_y_array - np.max(data_y_array) / 2))] - mean)
    			norm = (np.max(data_y_array) - np.min(data_y_array)) / gaussian(mean, 1.0, 0.0, mean, std)
    			floor = np.min(data_y_array)

    			try:

        			[norm, floor, mean, std], covariance = \
            			curve_fit(gaussian, data_x_array, data_y_array, p0=[norm, floor, mean, std])

        			return norm, floor, mean, std

    			except RuntimeError:

        			print('fit_gaussian: could not find a Gaussian')

			        return np.nan, np.nan, np.nan, np.nan




		########################################################################

	def distribution1d(data_array, step=None, binning=None):

    			binning = int(binning)

    			data_array = np.array(data_array).flatten()
    			data_array = np.sort(data_array)

    			if binning:
        			start = data_array.size - (data_array.size // binning) * binning
        			data_array = np.mean(np.reshape(data_array[start:], (data_array.size // binning, binning)), 1)

    			if not step:
        			step = np.sqrt(np.median((data_array - np.median(data_array)) ** 2)) / 5.0

    			min_value = min(data_array)
    			max_value = max(data_array)
    			bins_number = int((max_value - min_value) / step) + 2

			bins = min_value + step / 2. + np.arange(bins_number) * step
			bins_i = bins - step / 2
			bins_f = bins + step / 2
			counts = np.bincount(np.int_((data_array - min_value) / step))
			counts = np.insert(counts, len(counts), np.zeros(bins_number - len(counts)))

			return bins_i, bins_f, counts

		########################################################################
		
	def fit_2d_gauss_point(data_array, predicted_x_mean, predicted_y_mean, search_window):

    			def function_2d_gauss(xy_array, model_norm, model_floor,
                        	  model_x_mean, model_y_mean, model_x_std, model_y_std, model_theta):

		        	x_array, y_array = xy_array

		        	a = (np.cos(model_theta) ** 2) / (2 * model_x_std ** 2) + (np.sin(model_theta) ** 2) / (2 * model_y_std ** 2)
		        	b = -(np.sin(2 * model_theta)) / (4 * model_x_std ** 2) + (np.sin(2 * model_theta)) / (4 * model_y_std ** 2)
		        	c = (np.sin(model_theta) ** 2) / (2 * model_x_std ** 2) + (np.cos(model_theta) ** 2) / (2 * model_y_std ** 2)

		        	return (model_floor + model_norm * np.exp(- (a * ((x_array - model_x_mean) ** 2)
                                	                  + 2.0 * b * (x_array - model_x_mean) * (y_array - model_y_mean)
                                	                  + c * ((y_array - model_y_mean) ** 2)))).flatten()

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
            			curve_fit(function_2d_gauss,
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
            				found = False

    			except RuntimeError:

			        found = False

			if found:
			        return norm, floor, x_mean, y_mean, x_std, y_std

    			else:
        			print('fit_2d_gauss: could not find a 2D Gaussian')
        			return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


		################################################################################


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


class align():

    reduced_files = Frame.objects.filter(campaign_id=1,observation_type='S')       #####dDONT KNOW WHERE TO GET THESE FROM
	
	#before it's got something about accessing alignment true/false from write_log, tis is presumably something taken from user input...template stuff

    science = glob.glob('{0}{1}*.f*t*'.format(reduction_directory, os.sep))  ##what are reduction files saved as in model?
    

    science.sort()     #DO WE NEED A SORT IF READING IN FROM MODEL?

    fits = pf.open(science[0], memmap=False)       #CANT AVOID OPENING FITS HERE

    shift_tolerance = int(max(len(fits[1].data), len(fits[1].data[0])) * (shift_tolerance / 100.0))
    y_length, x_length = fits[1].data.shape

    centroids = []
    std_limit = 5.0
    while len(centroids) == 0 and std_limit >= 1.0:
        centroids = functs_align.find_centroids(fits[1].data, mean=MEAN, std=STD,                       #CARRY IN JUST PUTTING NAMES OF FITS HEADERS HERE
                                   std_limit=std_limit, burn_limit=7.0 * burn_limit / 8, star_std=2)
        std_limit -= 1.0
    calibration_stars = [[-pp[3], pp[1], pp[2], pp[0]] for pp in centroids]                     

    new_star_std = []
    for calibration_centroid in calibration_stars[:int(0.1 * len(calibration_stars))]:
        norm, floor, x_mean, y_mean, x_sigma, y_sigma = \
            functs_align.fit_2d_gauss_point(fits[1].data,                                    ####FIND WHERE THESE R USED
                               predicted_x_mean=calibration_centroid[1],
                               predicted_y_mean=calibration_centroid[2],
                               search_window=2 * star_std)
        if not np.isnan(x_mean * y_mean):
            new_star_std.append(x_sigma)
    star_std = max(1, int(np.median(new_star_std)) - 1)
    ######write_log('alignment', star_std, 'star_std')              update star_std in database

    centroids = []
    std_limit = 5.0
    while len(centroids) == 0 and std_limit >= 1.0:
        centroids = functs_align.find_centroids(fits[1].data, mean=fits[1].header[mean_key], std=fits[1].header[std_key],
                                   std_limit=std_limit, burn_limit=7.0 * burn_limit / 8, star_std=2 * star_std)
        std_limit -= 1.0
    calibration_stars = [[-pp[3], pp[1], pp[2], pp[0]] for pp in centroids]

    x_ref_position = np.nan
    y_ref_position = np.nan
    while np.isnan(x_ref_position * y_ref_position):
        norm, floor, x_mean, y_mean, x_sigma, y_sigma = functs_align.fit_2d_gauss_point(
            fits[1].data,
            predicted_x_mean=calibration_stars[0][1], predicted_y_mean=calibration_stars[0][2],
            search_window=2 * star_std)
        x_ref_position, y_ref_position = x_mean, y_mean
        del calibration_stars[0]

    centroids = functs_align.find_centroids(fits[1].data, mean=fits[1].header[mean_key], std=fits[1].header[std_key],
                               std_limit=3.0, burn_limit=7.0 * burn_limit / 8, star_std=star_std)
    calibration_stars = [[-pp[3], pp[1], pp[2], pp[0]] for pp in centroids]
    calibration_stars.sort()

    # take the rest as calibration stars and calculate their polar coordinates relatively to the first
    calibration_stars_polar = []
    for calibration_star in calibration_stars[:100]:
        norm, floor, x_mean, y_mean, x_sigma, y_sigma = \
            functs_align.fit_2d_gauss_point(fits[1].data,
                               predicted_x_mean=calibration_star[1],
                               predicted_y_mean=calibration_star[2],
                               search_window=2 * star_std)
        x_position, y_position = x_mean, y_mean
        r_position, u_position = functs_align.cartesian_to_polar(x_position, y_position, x_ref_position, y_ref_position)      
        if not np.isnan(x_position * y_position):
            calibration_stars_polar.append([r_position, u_position])

    x0, y0, u0, comparisons = x_ref_position, y_ref_position, 0, calibration_stars_polar
    fits.close()
