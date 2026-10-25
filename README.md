# LD1 LF Superheat Control Model

## Overview

This project was developed during my AI/ML internship at Tata Steel Prashikshan. The objective was to build a machine learning model for predicting Ladle Furnace (LF) temperature behavior and assisting operators in maintaining the desired superheat range during steelmaking operations.

## Problem Statement

In steel manufacturing, maintaining optimal superheat is critical for product quality, process stability, and operational efficiency. This project aimed to analyze historical furnace data and develop a predictive model to support temperature control and decision-making.

## Dataset

* Industrial Ladle Furnace production data
* 3,300+ operational records collected over 6 months
* Multiple process parameters including temperatures, durations, and ladle-related variables

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest Regression

## Project Workflow

1. Data Collection and Understanding
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Exploratory Data Analysis (EDA)
5. Model Training and Evaluation
6. Temperature Prediction and Superheat Analysis

## Results

* Trained and evaluated machine learning models on industrial furnace data for temperature prediction.
* Achieved approximately **95% R²** on a focused 2-month dataset (1,800+ records) with a richer feature set.
* Achieved approximately **55% R²** on a broader 6-month dataset (3,300+ records) with fewer available features, highlighting the impact of feature quality and process variability on model performance.
* Identified key process parameters affecting superheat control.
* Developed a predictive workflow to assist in determining target furnace temperatures and maintaining the desired superheat range (30–45°C).

## Key Learnings

* Working with real-world industrial datasets
* Feature engineering and feature selection
* Regression modeling using Random Forest
* Model evaluation using R² and error metrics
* Understanding data variability, model generalization, and industrial process behavior

## Internship

Developed as part of the AI/ML Internship Program at Tata Steel Prashikshan.
