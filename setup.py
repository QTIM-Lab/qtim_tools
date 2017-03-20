from setuptools import setup, find_packages
from codecs import open
from os import path

# with open(path.join('./', 'README.md'), encoding='utf-8') as f:
    # long_description = f.read()

setup(
  name = 'qtim_tools',
  # packages = ['qtim_tools'], # this must be the same as the name above
  version = '0.1.9',
  description = 'A library for medical imaging analysis, with an emphasis on MRI, machine learning, and neuroimaging. Created by the Quantiative Tumor Imaging Lab at the Martinos Center (Harvard-MIT Program in Health, Sciences, and Technology / Massachussets General Hospital)',
  packages=find_packages(),
  author = 'Andrew Beers',
  author_email = 'abeers@mgh.harvard.edu',
  url = 'https://github.com/QTIM-Lab/qtim_tools', # use the URL to the github repo
  download_url = 'https://github.com/QTIM-Lab/qtim_tools/tarball/0.1.9', # I'll explain this in a second
  keywords = ['neuroimaging', 'niftis', 'nifti','mri','dce','dsc','ktrans','ve','tofts','machine learning','vision','texture','learning'], # arbitrary keywords
  classifiers = [],
)