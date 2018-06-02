from django.db import models


class Target(models.Model):

    def __str__(self):
        return self.object_name

    object_name = models.CharField(max_length=200)
    object_RA = models.CharField(max_length=12)
    object_DEC = models.CharField(max_length=10)


class Frame(models.Model):

    def __str__(self):
        return str(self.observation_image.name.strip('media'))

    campaign_id = models.IntegerField()
    observation_image = models.ImageField(upload_to='media')
    observation_object = models.ForeignKey(Target, on_delete=models.CASCADE)
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


class Dataset(models.Model):

    def __str__(self):
        return str(self.name)

    name = models.TextField(default='placeholder')
    data = models.ManyToManyField(Frame)


class PhotometrySource(models.Model):

    def __str__(self):
        return str(self.observation_target)

    observation_target = models.ForeignKey(Target, on_delete=models.CASCADE)
    sourcePosX = models.FloatField(default=0)
    sourcePosY = models.FloatField(default=0)
    sourceCount = models.IntegerField(default=0)