from django.contrib import admin
from .models import Dataset,Frame,PhotometrySource,Target

admin.site.register(Dataset)
admin.site.register(Frame)
admin.site.register(PhotometrySource)
admin.site.register(Target)