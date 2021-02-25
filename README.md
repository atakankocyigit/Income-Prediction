# Income-Prediction
Performance comparison was made by using 6 different machine learning models in the project, where the income data set was used. The following libraries must be installed in order to run the project.
- library(e1071)
- library(caTools)
- library(tree)
- library(caret)
- library(rattle)
- library(ggplot2)
- library(maptree)
- library(ggcorrplot)
- library(Amelia)
- library(yardstick)
- library(visdat)

## DETAILS OF DATASET
The income dataset has 43,957 incidences and 14 descriptive features. The columns as known descriptive features are described in details below;
-	“Age” is the age of every individual.

-	“Workclass” is a term representing the employment status of an individual in the dataset. Private, Self employment, Government, Without payment, Never worked etc.

-	“Fnlwgt” means final weight and it is the number of people the census believes the entry represents.

-	“Education” represents the highest level of education achieved by an individual. Bachelors, Some college, 11th, HS grad., Profschool etc.

-	“Educational.num” is the highest level of education achieved, which is the numerical equivalent of the descriptive feature of education.

-	“Marital.status” represents the marital status of an individual. Married civ spouse refers to a civilian spouse, while Married AF spouse is a spouse in the Armed Forces.

-	“Occupation” are professions that belong to each individual. Tech-support is one of the occupations included in the dataset.

-	“Relationship” represents what the individual is relative to others as a family relations.

-	“Race” is a grouping in which people can be divided according to shared distinct physical characteristics. White, Black, Asian, Indian and Others.

-	“Gender” is a sex of the individual.

-	“Capital.gain” is capital gain annually for each individual.
 
-	“Capital.loss” is capital loss annually for each individual.

-	“Hours.per.week” is the hours of work per week of the individual.

-	“Native.country” represents the individual’s country of origin.

## PRE-PROCESS
Before model training, preprocessing should be done on the dataset so that the models do not fail. This process is performed on the rows with missing data, and then the numerical data must be brought to a certain standard (z-score). Undersampling and parameter tuning has been done.

## LICENSE
MIT LICENSE
