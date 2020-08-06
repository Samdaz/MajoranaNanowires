import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MajoranaNanowires-Quantum_Simulation_Package",
    version="1.0",
    author="Samuel D. Escribano",
    author_email="samuel.diazes@gmail.com",
    description="A numerical package to model and simulate Majorana nanowire devices and other related semiconductor/superconductor heterostructures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Samdaz/MajoranaNanowires",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.6',
)
