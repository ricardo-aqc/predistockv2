from django.apps import AppConfig


class MyappConfig(AppConfig):
    defaultAutoField = 'django.db.models.BigAutoField'
    name = 'myapp'
