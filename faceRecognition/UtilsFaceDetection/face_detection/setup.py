from setuptools import setup, find_packages

def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="face_detection",
    version="0.0.1",
    description="Collection of functions for face detection",
    long_description=readme(),
    classifiers=[
        "License :: OSI Approved :: MIT Licence",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],

    keywords="",
    url="https://gitlab.algolook.com/development/UtilsFaceDetection",
    author="Fernando Herrera",
    author_email="fernandoj.herrera@softtek.com",
    license="MIT",

    python_requires=">=3.5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "nose==1.3.7",
        "coverage==4.5.4",
        "pytest==5.1.1",
        "numpy==1.16.2",
        # OpenCV is installed in the custum containers
        # The nvidia container has a special version of OpenCV
        # that has cuda enabled
        #"opencv_contrib_python==4.2.0.32", 
        "imutils==0.5.3",
        "dlib==19.18.0",
        "utils_general==0.0.1",
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        #"console_scripts": ["start-api=api:main"]
    }
)
