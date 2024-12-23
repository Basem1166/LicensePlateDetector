# License Plate Detector

## Introduction

> This is the project of the Image Processing academic elective course `CMPS446` in Cairo University - Faculty of Engineering - Credit Hours System - Communication and Computer program
>
> This project is about applying image processing techniques to localize the license plate in an image and apply OCR to get the license plate number.

***

## Used Technologies

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"> <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white"> <img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white"> <img src="https://img.shields.io/badge/scikit--image-%23F7931E.svg?style=for-the-badge&logo=scipy&logoColor=white">



## Pipeline

> Our implementation was first strongly inspired by this research paper ([A New Approach for License Plate Detection and Localization Between Reality and Applicability](https://www.researchgate.net/publication/283758434_A_New_Approach_for_License_Plate_Detection_and_Localization_Between_Reality_and_Applicability)). However, we have read other papers and started to mix between the approaches that we have found until we decided on the following pipeline.

### Preprocessing
1.	Bilateral Filter
2.	CLAHE
3.	Image Binarization (Formula in the research paper)

### License Plate Detection
4.	Getting Initial ROI Regions.
5.	Finding final ROI Regions.
6.	Updating the edge powers
7.	Selecting an ROI Candidate
8.	Locating License Plate Columns
9.	A Growing Window Filter.

### License Plate Reading
10.	Gabor transform 
11.	 OCRCharacters,
12.	get fill colour 
13.	 extract plate
14. DNN


***

## Usage

```bash
python -m streamlit run main.py
```

