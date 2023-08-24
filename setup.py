import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='sdelearn',
    version='0.1.6',
    packages=['sdelearn'],
    url='https://github.com/fiafrate/sdelearn',
    license='MIT',
    author='Francesco Iafrate',
    author_email='francesco.iafrate@uniroma1.it',
    description='SDElearn: a Python framework for Stochastic Differential Equations modeling',
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "scipy", "sympy", "pandas", "matplotlib"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ],
    keywords=["stochastic", "differential", "equations", "statistical learning", "inference"]
)
