import os
from setuptools import setup, find_packages

# 1. Define the directory where setup.py is located
directory = os.path.abspath(os.path.dirname(__file__))

# 2. Read the README from the PARENT directory (..)
with open(os.path.join(directory, '..', 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tirank',
    version='1.0.2',  # Bump version to force a new build
    author='Lenis Lin',
    author_email='727682308@qq.com',
    license='MIT',
    description='A comprehensive analysis tool for prioritizing phenotypic niches in tumor microenvironment.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/LenisLin/TiRank',
    
    # 3. CRITICAL: Tell setuptools to look for packages in the parent directory
    packages=find_packages(where='..'), 
    package_dir={'': '..'},

    install_requires=[
        'numpy>=1.22.0,<2.0.0',
        'pandas>=1.5.0',
        'scipy>=1.8.0',
        'statsmodels>=0.14.0',
        'scikit-learn>=1.0.0',
        'openpyxl',
        'python-igraph', 

        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'timm>=0.5.4',

        'scanpy>=1.9.0',
        'leidenalg',
        'gseapy>=1.0.0',
        'lifelines>=0.27.0',
        'imbalanced-learn>=0.11.0',

        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'pillow>=9.0.0',

        'optuna>=3.0.0',
        'click',
        'dash>=2.0.0',
        'dash-bootstrap-components>=1.0.0',
        'snakemake',
    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.9',
)