# Generated by Django 2.0.1 on 2018-05-30 12:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hops', '0012_auto_20180530_1353'),
    ]

    operations = [
        migrations.AddField(
            model_name='frame',
            name='observation_filename',
            field=models.TextField(default='filename', max_length=50),
            preserve_default=False,
        ),
    ]
