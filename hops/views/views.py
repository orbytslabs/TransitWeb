from astropy.io import fits

from django.conf import settings
from django.template import loader
from django.http import HttpResponse

import matplotlib.pyplot as plt
from ..models import Dataset,Frame

import os

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
        fits_img_path = Frame.objects.filter(observation_object__object_name=selected_dataset.name).first().observation_image.url
        data = fits.getdata(fits_img_path)
        image_path = os.path.join(settings.BASE_DIR, 'static/{}.png'.format(fits_img_path.strip('/media/.fit')))
        plt.imshow(data, cmap='gnuplot',origin='lower')
        plt.axis('off')
        plt.savefig(image_path)
    else:
        raise ValueError('A very bad thing happened.')

    template = loader.get_template('hops/source-selection.html')
    context = {
        'selected_dataset': selected_dataset,
        'image_path': '/static/{}.png'.format(fits_img_path.strip('/media/.fit'))
    }
    return HttpResponse(template.render(context, request))