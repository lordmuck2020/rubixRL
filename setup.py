from setuptools import setup, find_packages

setup(
    name="rubixRL",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.19.0",
    ],
    author="JKRH",
    author_email="j.hancock354@gmail.com",
    description="A Gymnasium-based Reinforcement Learning Environment for Rubik's Cube",
    keywords="reinforcement-learning, gymnasium, rubiks-cube, artificial-intelligence",
    url="https://github.com/lordmuck2020/rubixRL",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
