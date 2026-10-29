"""gunicorn entry point (App Service startup: gunicorn ... wsgi:app).

A plain module:attribute keeps the startup command free of quotes and
parentheses, which App Service's startup script can't pass through eval.
"""
from app import create_app

app = create_app()
