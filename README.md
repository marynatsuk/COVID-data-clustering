# COVID Data Clustering Project Overview

This project contains an analysis of data on [COVID-19](https://ourworldindata.org/coronavirus) provided by OWID and its correlation on [World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness).
The main aim of this project was to examine and analyse different methods of clustering and their results. Data from these datasets was clustered using K-means, DBSCAN and hierarchical clustering.
The following images contain the results of the analysis based on specified parameters of chosen datasets.

## K-means clustering
![image](https://user-images.githubusercontent.com/73252923/217788281-d58f8459-e27a-4565-8471-d5294747ab28.png)

## DBSCAN clustering
![image](https://user-images.githubusercontent.com/73252923/217788445-5386317d-bfeb-4996-9458-2921e946af21.png)

## Hierarchical clustering
![image](https://user-images.githubusercontent.com/73252923/217788532-219fb3c4-4735-4a9f-902c-953e073a2eb0.png)

# Analysis results
In the clustering process countries were divided into the next 4 groups:
  1. Countries with developed economies, high levels of life and happy citizens with __high__ levels of COVID-19 concerns.
  2. Countries with a medium level of life with COVID-19 concerns above average.
  3. Countries with a medium level of life with COVID-19 concerns below average.
  4. Countries with not developed economies and low levels of life with __low__ amount of COVID-19 cases.

Since countries with developed economies have high incidence rates, it can be concluded that countries with better living standards have higher statistics, as these countries can collect information on disease transmission more quickly and efficiently. It can also be noted that countries with a low quality of life are likely to have low morbidity and mortality statistics because COVID testing has not been performed on a large scale in these countries, so many cases of diseases
