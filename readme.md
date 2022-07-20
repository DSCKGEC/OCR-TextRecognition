# OCR Text Recognition

[![Contributors](https://img.shields.io/github/contributors/DSCKGEC/OCR-TextRecognition.svg)](https://github.com/dsckgec/project-template/graphs/contributors) [![Forks](https://img.shields.io/github/forks/DSCKGEC/OCR-TextRecognition.svg)](https://github.com/dsckgec/project-template/network/members) [![Issues](https://img.shields.io/github/issues/DSCKGEC/OCR-TextRecognition)](https://github.com/dsckgec/project-template/issues) [![Pull Request](https://img.shields.io/github/issues-pr-closed-raw/DSCKGEC/OCR-TextRecognition)](https://github.com/dsckgec/project-template/pulls)


Recognizes text from any images and prints them.

## Contents 

- [OCR Text Recognition](#ocr-text-recognition)
  - [Contents](#contents)
  - [Description](#description)
    - [What's the problem?](#whats-the-problem)
    - [How can this project help?](#how-can-this-project-help)
    - [The idea](#the-idea)
  - [Project structure](#project-structure)
  - [Project roadmap](#project-roadmap)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installing](#installing)
  - [Live demo](#live-demo)
  - [Built with](#built-with)
  - [Contributing](#contributing)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Description
- This Python Script performs text detection using OpenCV’s EAST text detector, a highly accurate deep learning text detector used to detect text in natural scene images.Once we have detected the text regions with OpenCV, we’ll then extract each of the text ROIs and pass them into Tesseract, enabling us to build an entire OpenCV OCR pipeline!
### What's the problem?

### How can this project help?
- This project can help you recognize text on an image. OCR technology solves the problem by converting text images into text data that can be analyzed by other business software. You can then use the data to conduct analytics, streamline operations, automate processes, and improve productivity.It is based on OpenCV, enabling us to apply deep learning.

### The idea
- 
## Project structure

```
```

## Project roadmap

The project currently does the following things.

- Text Recognition
- Text Detection
- Prints Recognized Text

See below for our future steps.

- Converting Text to Audio
- Language Conversion

## Getting started
- In order to perform OpenCV OCR text recognition, we will need to install various python packages such as Tesseract v4 which includes a highly accurate deep learning-based model for text recognition, OpenCV and other required packages.

### Prerequisites
- Tesseract v4: It includes a highly accurate deep learning-based model for text recognition.
- OpenCV: To run this script you’ll need OpenCV installed. Version 3.4.2 or better is required.
- frozen_east_text_recognition.pb: The EAST text detector. This CNN is pre-trained for text detection and ready to go.
- imutils: This package will be used for non-maxima suppression.
- argparse
- numpy

### Installing

A step by step series of examples that tell you how to get a development env running.
Before getting on with the installations, make sure to download the frozen_east_text_recognition.pb from [link](https://drive.google.com/drive/folders/1Fta1KanMe7Pv6u9beUk8llbca77LOeax?usp=sharing).

Installing Tesseract v4:
- In Windows: Under python Terminal type: pip install tesseract.
Installing OpenCV:
- In Windows: Under python Terminal type: pip install opencv-python
- frozen_east_text_recognition.pb: It can be downloaded from the project as it is already included(must be kept in the same folder as the script).
- imutils: Under python Terminal type: pip install imutils
- argparse: Under python Terminal type: pip install argparse
- numpy: Under python Terminal type: pip install numpy

## Live demo


## Built with

- using OpenCV’s EAST text detector, a highly accurate deep learning text detector used to detect text in natural scene images.
- Once we have detected the text regions with OpenCV, we’ll then extract each of the text ROIs and pass them into Tesseract, enabling us to build an entire OpenCV OCR pipeline!

## Contributing

Please read [contributing.md](contributing.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

<a href="https://github.com/DSCKGEC/project-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DSCKGEC/OCR-TextRecognition" />
</a>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- 
