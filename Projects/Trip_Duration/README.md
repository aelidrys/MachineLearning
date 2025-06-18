# Trip Duration Prediction

## Definition from kaggle
#### In this competition, Kaggle is challenging you to build a model that predicts the total ride duration of taxi trips in New York City. Your primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables. 

## Project Overview
- ### Understanding Features or columns
    - #### `id`: a unique identifier for each trip
    - #### `vendor_id` : a code indicating the provider associated with the trip record
    - #### `pickup_datetime` : date and time when the meter was engaged
    - #### `dropoff_datetime` : date and time when the meter was disengaged
    - #### `passenger_count` : the number of passengers in the vehicle (driver entered value)
    - #### `pickup_longitude` : the longitude where the meter was engaged
     - #### `pickup_latitude` : the latitude where the meter was engaged
    - #### `dropoff_longitude` : the longitude where the meter was disengaged
    - #### `dropoff_latitude` : the latitude where the meter was disengaged
    - #### `store_and_fwd_flag` : This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server : Y=store and forward; N=not a store and forward trip
- ### Data Cleaning
    - #### check bout data problems like missing values, outliers, categorical data.
    - #### if there is one of them or more begin in handling and traitement

- ### Feature Engeneering
    - #### select features
    - #### add features: 
        - #### add distance column instead of coordinates columns (latitude, longitude)
        - #### convert pickup_datetime column to pickup_hour and pickup_month and pickup_weekday

- ### Training
    - #### scale data by using `MiniMax` scaler
    - #### generate more features with `PolynomialFeatures`
    - #### fit `Ridge` model
    - #### report train and test performence

---
## Performence
### train R2_score: 63.11%
### test R2_score: 59.78%