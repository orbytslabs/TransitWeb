from django.db import models

class Observation(models.Model):
    observation_date = models.DateField()
    observation_time = models.TimeField()
    exposure_time = models.IntegerField(default=0)

class Object(models.Model):    
    object_name = models.CharField(max_length=200)
    object_ra = models.CharField(max_length=12)
    object_dec = models.CharField(max_length=10)
