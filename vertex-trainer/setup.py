from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "pyyaml",
    "scikit-learn==0.21.3",
    "joblib==0.13.2",
    "google-cloud-storage",
    "google-cloud-bigquery",
    "gcsfs",
]

setup(
    name="trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),  # Automatically find packages within this directory or below.
    include_package_data=True,  # if packages include any data files, those will be packed together.
    description="Classification training titanic survivors prediction model",
)
