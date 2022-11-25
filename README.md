# CoE-197M-Projects
Course Title: **CoE 197M Foundations of Machine Learning** \
Section: **M-THY** \
Instructor: **Rowel Atienza**

## Assignment 1 - Removing Projective Distortion on Images

Packages required:
* tkinter
* pillow
* numpy

Install requirements:
```
pip install -r requirements.txt
```

Usage:
```
python main.py
```

## Assignment 2 - Polynomial Solver using SGD in TinyGrad
Objective:
* SGD is a useful algorithm with many applications. In this assignment, we will use SGD in the TinyGrad framework as a polynomial solver - to find the degree and coefficients. \
Install requirements:
```
pip install -r requirements.txt
```
Usage: 
```
python3 solver.py
```
The solver will use `data_train.csv` to estimate the degree and coefficients of a polynomial. To test the generalization of the learned function, it should have small test error on `data_test.csv`.
