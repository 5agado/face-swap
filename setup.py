from setuptools import setup, find_packages


setup(
    name="face_swap",
    version="0.1.0",
    author="Alex Martinelli",
    description="Face swapping tools and utils",
    license="Apache 2.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': ['extract=face_swap.extract:main',
                            'deep_swap=face_swap.deep_swap:main',
                            'base_swap=face_swap.base_swap:main'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'opencv-python',
        'scipy',
        'tensorflow',
        'keras',
        'scikit-image',
        'tqdm',
        'dlib',
        'mtcnn', #https://github.com/ipazc/mtcnn
        'data-science-learning==0.1.0',
    ],
    dependency_links=[
        'https://github.com/5agado/data-science-learning/tarball/master#egg=data-science-learning-0.1.0'
    ],
)