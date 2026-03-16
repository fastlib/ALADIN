import os
import warnings

aladin_model_folder = os.environ.get('aladin_models')
aladin_cache_folder = os.environ.get('aladin_cache')

if aladin_model_folder is None:
    raise ValueError("Environment variable 'aladin_models' is not set. Please set it to the path where ALADIN models are stored.")
if aladin_cache_folder is None:
    warnings.warn("Environment variable 'aladin_cache' is not set. This is not a breaking error, but please set this path if you want to use ALADIN's caching capabilities.")
