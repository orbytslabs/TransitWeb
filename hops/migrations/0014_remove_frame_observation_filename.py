# Generated by Django 2.0.1 on 2018-05-30 13:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('hops', '0013_frame_observation_filename'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='frame',
            name='observation_filename',
        ),
    ]
