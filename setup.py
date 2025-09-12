from setuptools import setup, find_packages

# This setup.py is maintained for editable installs compatibility
# Main configuration is in pyproject.toml

setup(
    name='vncNet',
    packages=find_packages(),
    package_dir={'': '.'},
    include_package_data=True,
)