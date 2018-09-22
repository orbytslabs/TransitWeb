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
    image_width = models.IntegerField(default=844)
    image_height = models.IntegerField(default=644)
    observation_object = models.ForeignKey(Target, on_delete=models.CASCADE)
    observation_date = models.DateField()
    observation_start_time = models.DateTimeField()
    observation_end_time = models.DateTimeField()
    exposure_duration = models.IntegerField(default=0)
    aperture_size = models.IntegerField(default=7)
    align_x0 = models.IntegerField(default=0)
    align_y0 = models.IntegerField(default=0)
    align_u0 = models.IntegerField(default=0)
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
    image_height = models.IntegerField(default=0)
    image_width = models.IntegerField(default=0)
    data = models.ManyToManyField(Frame)


class PhotometrySource(models.Model):

    def __str__(self):
        return str(self.observation_target)

    observation_target = models.ForeignKey(Target, on_delete=models.CASCADE)
    source_pos_x = models.FloatField(default=0)
    source_pos_y = models.FloatField(default=0)
    source_pos_r = models.FloatField(default=0)
    source_pos_u = models.FloatField(default=0)
    source_type = models.CharField(default='target', max_length=10)
    aperture_size = models.IntegerField(default=7)