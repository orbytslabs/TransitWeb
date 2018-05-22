from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astropy.io import fits

from django.shortcuts import render

from ..models import *

# from .hops_basics import *

# class objectDetails(filename):
#     def __init__(self,targetname,ra,dec):
#         self.targetname = targetname
#         self.ra = ra
#         self.dec = dec
        
# class datasetdetails():
#     """
    
#     """
#     def __init__(self,obstype,exposure_time,time,date,im_filter):
#         self.obstype = obstype    #string standard (S) bias (B) flat (F) or dark (D)
#         self.exposure_time = exposure_time                           #NEED TO WRITE SOME CODE THAT GENERATES THIS FROM FITS FILE AND 'ADDS' IT AS AN ARGUMENT TO THE CREATION OF OBJECT
#         self.date = date
#         self.im_filter = im_filter
#         self.obsid = obsid
        
class reductionFiles():
    """
    Contains 3 methods:
        create_master_bias: creates a master bias image
        create_master_dark: creates a master dark image
        create_master_flat: creates a master flat image
    """
      
    def create_master_bias(bias_files):
        """
        Creates a master bias
        """
        bias_frames = []
        if len(bias_files) > 0:    
            for bias_file in glob.glob('*{0}*.f*t*'.format(bias_files)):
                fits = pf.open(bias_file, memmap=False)
                try:
                    bias_frames.append(float(fits[0].header['BZERO']) + float(fits[0].header['BSCALE']) * fits[0].data)
                except KeyError:
                    bias_frames.append(fits[0].data)
                fits.close()

        if len(bias_frames) > 0:
            if master_bias_method == 'median':
                master_bias = np.median(bias_frames, 0)
            elif master_bias_method == 'mean':
                master_bias = np.mean(bias_frames, 0)
            else:
                master_bias = np.median(bias_frames, 0)
        else:
            master_bias = 0.0
      
        return master_bias


    def create_master_dark(dark_files):
        """
        Creates a master dark
        """
        dark_frames = []
        if len(str(dark_files)) > 0:
            for dark_file in glob.glob('*{0}*.f*t*'.format(dark_files)):
                fits = pf.open(dark_file, memmap=False)
                try:
                    dark_frame = float(fits[0].header['BZERO']) + float(fits[0].header['BSCALE']) * fits[0].data
                except KeyError:
                    dark_frame = fits[0].data
                dark_frames.append((dark_frame - master_bias) /im_details.exposure_time)   
                fits.close()

        if len(dark_frames) > 0:
            if master_dark_method == 'median':
                master_dark = np.median(dark_frames, 0)
            elif master_dark_method == 'mean':
                master_dark = np.mean(dark_frames, 0)
            else:
                master_dark = np.median(dark_frames, 0)
        else:
            master_dark = 0.0

        return master_dark


    def create_master_flat(flat_files):
        """
        Creates a master flat
        """
        flat_frames = []
        if len(str(flat_files)) > 0:
            for flat_file in glob.glob('*{0}*.f*t*'.format(flat_files)):
                fits = pf.open(flat_file, memmap=False)
                try:
                    flat_frame = float(fits[0].header['BZERO']) + float(fits[0].header['BSCALE']) * fits[0].data
                except KeyError:
                    flat_frame = fits[0].data
                flat_frames.append(flat_frame - master_bias - im_details.exposure_time * master_dark)        #####NEEED WAY OF ACTUALLY GETTING EXP TIME FROM FIT HEADER, DO THIS WHEN DEFINING IM_DETAILS
                fits.close()                                                                               

        if len(flat_frames) > 0:
            if master_flat_method == 'median':
                flat_frames = [ff / np.median(ff) for ff in flat_frames]
                master_flat = np.median(flat_frames, 0)
            elif master_flat_method == 'mean':
                master_flat = np.mean(flat_frames, 0)
            else:
                flat_frames = [ff / np.median(ff) for ff in flat_frames]
                master_flat = np.median(flat_frames, 0)
            master_flat = master_flat / np.median(master_flat)
        else:
            master_flat = 1.0
        
        return master_flat



    class observationReduction():

        def __init__(self):
            self.textx = []
            self.testy = []
            self.testz = []

        def reduce_observations(self, observation_files):
            """
            Reduces the observations

            Args
                observation_files (str) 
                    - string containing identifying name of observation files 
            """
            # observation_files = glob.glob('../*{0}*.f*t*'.format(observation_files))
            observation_files = Frame.objects.filter(campaign_id=1,observation_type='S')
            observation_files.sort()
            percent = 0
            lt0 = time.time()

            for counter, science_file in enumerate(observation_files):

                # correct it with master bias_files, master dark_files and master flat_files

                fits = pf.open(science_file, memmap=False)
                try:
                    data_frame = float(fits[0].header['BZERO']) + float(fits[0].header['BSCALE']) * fits[0].data
                except KeyError:
                    data_frame = fits[0].data
                fits[0].data = (data_frame - master_bias - fits[0].header[exposure_time_key] * master_dark) / master_flat     ###RENAME EVERY TIME EXPOSURE TIME KEY IS USED TO IM_DETAILS.EXPOSURE
                fits[0].header.set('BZERO', 0.0)
                fits[0].header.set('BSCALE', 1.0)

                norm, floor, mean, std = fit_distribution1d_gaussian(fits[0].data, binning=fits[0].data.size / bin_to)

                if np.isnan(norm):
                    mean = np.mean(fits[0].data)
                    std = np.std(fits[0].data)

           

                julian_date = (ephem.julian_date(float(ephem.Date(local_time))) +
                               im_details.exposure / (2.0 * 60.0 * 60.0 * 24.0))

                ra_target, dec_target = target_ra_dec.split()

                heliocentric_julian_date = jd_to_hjd(ra_target, dec_target, julian_date)

                self.testx.append(heliocentric_julian_date)
                self.testy.append(mean)
                self.testz.append(std)

                fits[0].header.set(mean_key, mean)
                fits[0].header.set(std_key, std)

                # write the new fits file

                if observation_date_key == observation_time_key:
                        local_time = fits[0].header[observation_date_key]
                        local_time = '{0}_'.format(local_time.replace('-', '_').replace('T', '_').replace(':', '_'))
                else:
                        local_time = '{0}_{1}_'.format(fits[0].header[observation_date_key].split('T')[0].replace('-', '_'),
                                                       fits[0].header[observation_time_key].replace(':', '_'))

                try:
                    hdu = pf.CompImageHDU(header=fits[0].header, data=fits[0].data)
                except:
                    hdu = pf.ImageHDU(header=fits[0].header, data=fits[0].data)
                hdu.writeto('{0}{1}{2}{3}{4}'.format(reduction_directory,
                                                     os.sep, reduction_prefix, local_time, science_file.split(os.sep)[-1]))

                if counter == 0:
                    ax.cla()
                    ax.imshow(fits[0].data[::2, ::2], origin='lower', cmap=cm.Greys_r,
                              vmin=fits[0].header[mean_key] + frame_low_std * fits[0].header[std_key],
                              vmax=fits[0].header[mean_key] + frame_upper_std * fits[0].header[std_key])
                    ax.axis('off')

                    canvas.show()

                fits.close()