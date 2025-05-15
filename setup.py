from setuptools import setup

setup(
    name="pygtp",
    version="0.4",
    description="Python library for Go Text Protocol controllers and engines",
    license="MIT",
    url="http://github.com/jtauber/gtp",
    author="James Tauber",
    author_email="jtauber@jtauber.com",
    py_modules=["gtp"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Utilities",
    ],
)