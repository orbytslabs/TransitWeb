from django.db import models

class Object(models.Model):
    object_name = models.CharField(max_length=200)
    object_ra = models.CharField(max_length=12)
    object_dec = models.CharField(max_length=10)

class Observation(models.Model):
    campaign_id = models.IntegerField()
    observation_filename = models.FileField()
    observation_object = models.ForeignKey(Object, on_delete=models.CASCADE)
    observation_date = models.DateField()
    observation_start_time = models.TimeField()
    observation_end_time = models.TimeField()
    exposure_duration = models.IntegerField(default=0)
    OBSERVATION_TYPES = (
        ('S', 'Standard'),
        ('B', 'bias'),
        ('F', 'flat'),
        ('D', 'dark'),
    )
    observation_type = models.CharField(
        max_length=1,
        choices=OBSERVATION_TYPES,
        default='S',
    )
    OBSERVATION_FILTERS = (
        ('R', 'R'),
        ('G', 'G'),
        ('B', 'B'),
        ('U', 'U'),
        ('I', 'I'),
        ('V', 'V'),
    )
    observation_filter = models.CharField(
        max_length=1,
        choices=OBSERVATION_FILTERS,
        default='I',
    )

