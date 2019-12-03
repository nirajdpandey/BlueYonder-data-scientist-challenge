![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg) ![Depfu](https://img.shields.io/depfu/depfu/example-ruby.svg) 
![Travis (.org)](https://img.shields.io/travis/:user/:repo.svg) [![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

### Clean-code-Challenge
The repository contains the solution of BlueYonder GmbH's challenge for data scientist position. Implementing a regression model on bike-sharing data-set to predict count of future rentals. The Code Quality matters a lot here therefore, PEP-8 was followed to make script more pythonic :)


This includes following - 
- [x] Correct user defined fuction naming
- [x] Choosing clear variable names
- [x] Helper for all the functions 
- [x] Spaces and punctiations as per PEP-8 standard 
- [x] Raising clear exceptions 
- [ ]  Etc.....



***
Table of Contents
=================
* Data-set description 
* Data Summary
* Feature Engineering
* Missing Value Analysis
* Correlation Analysis
* Visualizing Distribution Of Data
* Visualizing Count Vs (Month,Season,Hour,Weekday,Usertype)
* Fitting the model 
* Results
***
#### Data-set description 
Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these systems, people are able rent a bike from a one location and return it to a different place on an as-needed basis. Our target is to predict the remtal count given the independent variables
***
#### Data Summary

Here you can see what is inside the data
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/Screenshot%20from%202019-02-01%2015-37-37.png)
***
#### Simple Visualization Of Variables number Count
all 4 Seasons seem to have eqaul count
`1:spring`
`2:summer`
`3:fall`
`4:winter`
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/season_count.png)
***
It is quit obvious that there would be less holiday and more of normal day in 2 years of time. 
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/holiday-vs-count.png)
***
Working day is having sattistics. `remember` in plots `0:False`and `1:True`

![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/working_day-vs-count.png)

***
Weather Count is as follows 
* weather

  1: Clear, Few clouds, Partly cloudy, Partly cloudy
  
  2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  
  3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
  
  4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog


![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/weather-vs-count.png)
***
Which month had the highest demand 
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/month-vs-count.png)
***
Which was the peak hour for renting the bike
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/hour-vs-count.png)
***
What temperature was best preferred for the ride
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/temp-vs-count.png)
***


#### Feature Engineering
You see! the columns "season","holiday","workingday" and "weather" should be of "categorical" data type.But the current data type is "int" for those columns. Let us transform the dataset in the following ways so that we can get started up with our `EDA` (Exploratory Data Analysis). 
```python
categoryVariableList = ["weekday",
                        "month",
                        "season",
                        "weather",
                        "holiday",
                        "workingday"]

for var in categoryVariableList:
    data[var] = data[var].astype("category")
```
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/data_types.png)
***
### Missing Value Analysis
Let's see if there is any `missing` on `NA` values in the entire dataset. SO, we dont have any missing value in the dataset. Yeeey...!!
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/missing_values.png)

***
#### Correlation Analysis

To understand how a dependent variable is influenced by features (numerical) is to get a correlation matrix between them. Lets plot a correlation plot between "count" and ["temp","atemp","humidity","windspeed"].

![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/corelation_mat.png)

>temp and humidity features has got positive and negative correlation with count respectively.Although the correlation between them are not very prominent still the count variable has got little dependency on "temp" and "humidity".

>windspeed is not gonna be really useful numerical feature and it is visible from it correlation value with "count"

>"atemp" is variable is not taken into since "atemp" and "temp" has got strong correlation with each other. During model building any one of the variable has to be dropped since they will exhibit multicollinearity in the data.

>"Casual" and "Registered" are also not taken into account since they are leakage variables in nature and need to dropped during model building.
***
#### Visualizing Count Vs (Month,Season,Hour,Weekday)
Looking at the following plot we can get some useful information. 
![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/count%20vs%20xyz.png)

>It is quiet obvious that people tend to rent bike during summer season since it is really conducive to ride bike at that season.Therefore June, July and August has got relatively higher demand for bicycle.

>On weekdays more people tend to rent bicycle around 7AM-8AM and 5PM-6PM. As we mentioned earlier this can be attributed to regular school and office commuters.

>Above pattern is not observed on "Saturday" and "Sunday".More people tend to rent bicycle between 10AM and 4PM.
***

After few more feature selection (see BlueYonder.py) We are all set to go for chosing the right Machine Learning model and evaluate it's performance. 
#### How to run 
```
1. clone this repository
2. open cmd in the cloned repo
3. type >>> python Clean_Regression.py

```
Now you can see the top hundred prediction and the loss of the model. If you want to see the plot of error by a specific regression model please visit explore_data folder of this repository. 

![image](https://github.com/nirajdevpandey/Clean-code-Challenge/blob/master/plots/error.png)

Thanks a lot
