"""AWS Lambda entry point. Adapts the Lambda event format
(Function URL / API Gateway HTTP v2) to a WSGI request for Flask.
"""
from apig_wsgi import make_lambda_handler
from app import app

# binary_support=True lets multipart image uploads + binary JSON pass through
handler = make_lambda_handler(app, binary_support=True)
