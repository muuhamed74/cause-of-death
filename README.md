# cause-of-death


### Project overview

A cause of death  project involves the development and deployment of algorithms or models trained on relevant datasets to automatically classify and predict the specific factors or conditions leading to an individual's death,
It aims to analyze data and explore the most common causes of stroke that lead to death.

dataset.head(10)


![image](https://github.com/muuhamed74/cause-of-death/assets/140603240/a2ae2efc-cbfd-4069-b0bd-efc12516746d)



### Data sources 

cause of death : the primary dataset used for this analysis is "Cause of Death_Training Part.csv" file , Which contains the details and reasons for performing the high stroke.

### Tools

-jupyter for python  [Download here](https://www.anaconda.com/download)

### Data cleaning

in the intial data pre-processing & feature engineering phase ,we performed the following tasks:

-Outliers (pre-processing)

-Missing Values (pre-processing)

-encoding & scaling (pre-processing)

-feature aggregation (feature engineering)

-feature extraction  (feature engineering)


### Machine learning model

#### classificatioin

-naive bayes 

-logistic regression

-random forest

###visualization

sns.countplot(x='X11',hue='Target',data=data)#smoke status

![image](https://github.com/muuhamed74/cause-of-death/assets/140603240/24e38528-942e-47dc-9ed0-f6b0742e20b1)


sns.countplot(x='X2',hue='Target',data=data)#gender compare

![image](https://github.com/muuhamed74/cause-of-death/assets/140603240/05198fd5-4419-4342-912e-86c7a29ae79c)


sns.countplot(x='X4',hue='Target',data=data)#if patient has Haipatensonu 

![image](https://github.com/muuhamed74/cause-of-death/assets/140603240/0b45a207-f8fb-4c81-a5d7-17207dd797af)



