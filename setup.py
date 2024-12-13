from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="AMOFMS",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,


    author="Zhixuan Zhong",
    author_email="zhixuan@iccas.ac.cn",
    description="AMOFMS: An Automated Mapping and Optimization Framework for Multiscale Simulation",
    long_description=open('README.rst').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/JiangGroup/AMOFMS",
    license="MIT",

    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=requirements,
    # openbabel

    python_requires='>=3.9',

    entry_points={
        'console_scripts': [
            'TP_MARTINI2 = AMOFMS.DSGPM_TP_MARTINI2.CGPredictionFromDSGPM_TP:main',
            ],
        },
)
