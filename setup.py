from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='swinggen',
    version='0.0.2',
    description='',
    py_modules=['swinggen'],
    package_dir={'': 'src'},
    classifiers=["Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.6",
                 "Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Programming Language :: Python :: 3.10",
                 "Operating System :: OS Independent"],
    license_files=['LICENSE.txt',],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy >= 1.8", "matplotlib >= 3.0.0"],
    extras_require={
        "dev": ["pytest >=3.7", ],
    },
    url="https://github.com/JonasFovea/swing-trajectory-generator",
    author="Jonas Grube",
    author_email="jonas.grube@fh-bielefeld.de",
)
