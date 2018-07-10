from setuptools import setup, find_packages
from codecs import open
from os import path

# with open(path.join('./', 'README.md'), encoding='utf-8') as f:
    # long_description = f.read()

#with open('requirements.txt') as f:
#    required = f.read().splitlines()

setup(
  name = 'qtim_tools',
  # packages = ['qtim_tools'], # this must be the same as the name above
  version = '0.1.13',
  description = 'A library for medical imaging analysis, with an emphasis on MRI, machine learning, and neuroimaging. Created by the Quantiative Tumor Imaging Lab at the Martinos Center (Harvard-MIT Program in Health, Sciences, and Technology / Massachussets General Hospital)',
  packages = find_packages(),
  entry_points =  {
                  "console_scripts": ['qtim = qtim_tools.__main__:main'], 
                  },
  author = 'Andrew Beers',
  author_email = 'abeers@mgh.harvard.edu',
  url = 'https://github.com/QTIM-Lab/qtim_tools', # use the URL to the github repo
  download_url = 'https://github.com/QTIM-Lab/qtim_tools/tarball/0.1.13',
  keywords = ['neuroimaging', 'niftis', 'nifti','mri','dce','dsc','ktrans','ve','tofts','machine learning','vision','texture','learning'], # arbitrary keywords
  install_requires=['dicom','pynrrd','Pillow','configparser','pyyaml','beautifulsoup4','nibabel','scikit-image','matplotlib','scipy','numpy'],
  classifiers = [],
)