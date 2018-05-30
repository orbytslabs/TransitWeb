from django.contrib import admin
from .models import Dataset,Frame,Target

admin.site.register(Dataset)
admin.site.register(Frame)
admin.site.register(Target)