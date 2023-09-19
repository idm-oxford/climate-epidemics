from setuptools import setup, find_packages

setup(name='epiclim', 
    version='0.0', 
    packages=find_packages(include=['epiclim']),
    install_requires = ['matplotlib','numpy','pandas','scipy'])
