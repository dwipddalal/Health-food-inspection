# Violation Detection Code

## Overview

This repository contains code for detecting possible violations using two main approaches:

1. **Past Event Analysis**: In this part of the code, we analyze historical data to assess the possibility of a violation occurring. To do this, I used the 3 datasets, clubbed them into 1 dataset and then performed NN based on other classification-based models. To handle the class imbalance I used focal loss here.

2. **Sanitary Conditions Assessment**: The second part of the code focuses on evaluating sanitary conditions in specific areas. It checks whether the conditions in a given location meet certain criteria. If the conditions do not meet the required standards, it may indicate a potential violation of regulations or guidelines.

## Code Structure

The code is organized into two main sections:

### 1. Past Event Analysis

This script contains the code for analyzing historical data and assessing the likelihood of violations based on past events. It includes functions for data preprocessing, pattern recognition, and risk assessment.

### 2. Sanitary Conditions Assessment

 This script is responsible for evaluating the sanitary conditions in specific areas. It defines criteria and checks whether the conditions in a given location meet those criteria. Any deviations from the standards are flagged as potential violations.

## Getting Started

To use the code for violation detection, follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/your_username/violation-detection.git

