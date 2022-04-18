
# Build with
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-380/)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/static/v1?style=for-the-badge&message=Streamlit&color=FF4B4B&logo=Streamlit&logoColor=FFFFFF&label=)
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/posts/hitesh-chaudhari-0259ba14a_project-collaboration-content-activity-6921523549367095296-iEiW?utm_source=linkedin_share&utm_medium=member_desktop_web)

# AQI(Air Quality Index) Analysis ,Visualization and Forecasting.

A web App deployed on HEROKU CLOUD/Strealit CLoud ,that Analyse & visualize the Air Quality Data collected on hourly basis from 153 stations in india and than Forecast the AQI(Air Quality Index) for next 7 days, for every individual station.

Watch a short App flow video here 👇.

https://user-images.githubusercontent.com/72800256/163836662-ac229c24-bf3c-414f-837a-fb104be99958.mp4

You can try out this app [here.](https://share.streamlit.io/hitman-dev/indias-aqi/app.py)

## What is AQI & how is it calculated ?

AQI is defined as an overall scheme that transforms weighted values of individual air pollution
related parameters (SO2, CO, visibility, etc.) into a single number or set of numbers. The Air Quality
Index criteria is defined by the Central Pollution Control Board of India (CPCB), Delhi. CPCB air
quality standards necessitate for 12 parameters – PM 10, PM 2.5, NO2, SO2, CO, O3, NH3, Pb, Ni,
As, Benzo(a) pyrene, and Benzene. However, the selection of parameters primarily depends on AQI
objective(s),data availability, averaging period, monitoring frequency, and measurement methods.
While PM 10, PM 2.5, NO2, SO2, NH3 and Pb have 24-hourly as well annual average standards,
Ni, As, benzo(a)pyrene, and benzene have only annual standards and CO and O 3 have short-term
standards (01 and 08 hourly average). PM 10, PM 2.5, SO2, NO2, CO, and O 3 are measured on a
continuous basis at many air quality stations (including NH 3 at some stations), Pb, Ni, As,
Benzo(a)pyrene and NH 3, if monitored, use manual systems. To get an updated AQI at short time
intervals, ideally eight parameters (PM 10, PM 2.5, NO2, SO2, CO, O3, NH3, and Pb) for which,
short-term standards are prescribed should, be measured on a continuous basis.

## Project overview
In this project we have collected data from various sources like kaggle and API such as WAQI and OPENWEATHERMAP, performed EDA on the data and stored it on GitHub, then apply Forecasting model.

## API Used

Below are the links of API's used to collect AQI related information

- WAQI Api to collect location co-ordinates of the AQI monitoring stations in India.
  - Get Active stations [notebook.](https://github.com/hitman-dev/Indias-AQI/blob/master/notebooks/get_active_stations.ipynb)
  - API link :- 'https://aqicn.org/data-platform/token/'

- OpenWeathermap API to get concentration of the particulates from the 153 AQI monitoring stations in India based on the location co-ordinates gathered from WAQI API.
  - Get concentration of particulates from active stations [notebook.](https://github.com/hitman-dev/Indias-AQI/blob/master/notebooks/2020-2022_dataCollection.ipynb) 
  - API link :- 'https://openweathermap.org/api/air-pollution'
 
## Calculating AQI 

AQI is calculated with the help of [Guidelines](https://app.cpcbccr.com/ccr_docs/How_AQI_Calculated.pdf) given by [CPCB](https://cpcb.nic.in/index.php)(Central Pollution Contrl Board)
AQI bucketing is used to understand the air quality for a region which is based on the values of AQI. The range of AQI, colors assigned and its corresponding effects are assigned as per the CPCB guidelines and are as follows.

![Screenshot](images/AQI_Range.JPG)

The range of 24hr average of concentration of a individua particulate, colors assigned and its corresponding effects are also specified as per the CPCB guidelines and are as follows.

![Screenshot](images/concentration_range.JPG)

The Exploratory Data Analysis(EDA) and calculation of the AQI as specified by the above mention guidelines are shown i this [notebook](https://github.com/hitman-dev/Indias-AQI/blob/master/notebooks/2020-2022_history_data_processed.ipynb)

## AQI Forecasting

For forecasting we have used various algorithms and auto ML libraries but the best out was given by Pycaret auto ML library.
PyCaret is an Auto-ML library used for the building machine learning models. PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows.It is an end-to-end machine learning and model management tool that exponentially speeds up the
experiment cycle and makes you more productive.
You can get more information about PyCaret[here.](https://pycaret.org/)
So by using PyCaret Forecasting Algorithm, we made individual models for each station(total 153 ML models) which are loaded dynamically and gives the respective station forecasting for next 7 days.

```python
from pycaret.regression import *

data = df.groupby(['City','Date']).mean().reset_index()
data['Date'] = pd.to_datetime(data['Date'])
data['month'] = [i.month for i in data['Date']]
data['year'] = [i.year for i in data['Date']]
data['day_of_week'] = [i.dayofweek for i in data['Date']]
data['day_of_year'] = [i.dayofyear for i in data['Date']]

cities = data['City'].unique().tolist()

data=data.drop([ 'PM25', 'PM10', 'CO', 'NO2', 'NH3', 'O3', 'SO2'],axis=1)
data.set_index('Date',inplace=True)

dates = pd.date_range(start='2020-11-28', end = '2022-02-09', freq = 'D')
ma_df = pd.DataFrame()
ma_df['Date'] = dates
ma_df.set_index('Date',inplace=True)


for i in cities:
  subset = data[data['City'] == i]
  subset[f'{i}_MA'] = subset['AQI'].rolling(30).mean()
  subset.rename(columns = {'AQI': f'{i}_AQI'}, inplace = True)
  subset.drop(columns=['City','month','year','day_of_week','day_of_year'],inplace=True)
  ma_df = pd.merge(ma_df, subset, left_index=True, right_index=True)
  
  
all_ts = data['City'].unique()
all_results = []
final_model = {}

for i in all_ts:
    df_subset = data[data['City'] == i]
    # initialize setup from pycaret.regression
    s = setup(df_subset, target = 'AQI', train_size = 0.95, transform_target = True, remove_outliers = True, data_split_shuffle = False,
              fold_strategy = 'timeseries', fold = 5, ignore_features = ['City'], numeric_features = ['day_of_year', 'year'],
              categorical_features = ['month', 'day_of_week'], silent = True, verbose = False, session_id = 2022)
    # compare all models and select best one based on MAE
    best_model = compare_models(sort = 'MAE', verbose=True)
    
    # capture the compare result grid and store best model in list
    p = pull().iloc[0:1]
    p['City'] = str(i)
    all_results.append(p)
    
    # finalize model i.e. fit on entire data including test set
    f = finalize_model(best_model)
    
    # attach final model to a dictionary
    final_model[i] = f
    
    # save transformation pipeline and model as pickle file 
    save_model(f, model_name='trained_models/' + str(i), verbose=False)

```






