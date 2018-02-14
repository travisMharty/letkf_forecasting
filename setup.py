import os
try:

    from setuptools import setup, find_packages
except ImportError:
    raise RuntimeError('setuptools is required')


import versioneer


PACKAGE = 'letkf_forecasting'


SHORT_DESC = 'Make forecasts using the LETKF and satellite data'
AUTHOR = 'Travis Harty'


setup(
    name=PACKAGE,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=SHORT_DESC,
    author=AUTHOR,
    packages=find_packages(),
    include_package_data=True,
    scripts=[os.path.join('scripts', s) for s in os.listdir('scripts')]
)
