from setuptools import setup, find_packages

setup(
    name="ImageMathToTex",
    version="0.1",
    packages=find_packages(where='src'),  
    package_dir={'': 'src'}, 
    install_requires=[
        "texteller", "pytesseract", "ttkbootstrap", "opencv-python"
    ],
    entry_points={
        "console_scripts": [
            "imagemathtotex=main:main", 
        ],
    },
)
