
Global Power Plant Database Project

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
import joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import metrics
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# importing all necessary libraries
df = pd.read_csv("https://raw.githubusercontent.com/wri/global-power-plant-database/master/source_databases_csv/database_IND.csv")
df
country	country_long	name	gppd_idnr	capacity_mw	latitude	longitude	primary_fuel	other_fuel1	other_fuel2	...	geolocation_source	wepp_id	year_of_capacity_data	generation_gwh_2013	generation_gwh_2014	generation_gwh_2015	generation_gwh_2016	generation_gwh_2017	generation_data_source	estimated_generation_gwh
0	IND	India	ACME Solar Tower	WRI1020239	2.5	28.1839	73.2407	Solar	NaN	NaN	...	National Renewable Energy Laboratory	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	IND	India	ADITYA CEMENT WORKS	WRI1019881	98.0	24.7663	74.6090	Coal	NaN	NaN	...	WRI	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	IND	India	AES Saurashtra Windfarms	WRI1026669	39.2	21.9038	69.3732	Wind	NaN	NaN	...	WRI	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	IND	India	AGARTALA GT	IND0000001	135.0	23.8712	91.3602	Gas	NaN	NaN	...	WRI	NaN	2018.0	631.777928	617.789264	843.747000	886.004428	663.774500	Central Electricity Authority	NaN
4	IND	India	AKALTARA TPP	IND0000002	1800.0	21.9603	82.4091	Coal	Oil	NaN	...	WRI	NaN	2018.0	1668.290000	3035.550000	5916.370000	6243.000000	5385.579736	Central Electricity Authority	NaN
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
903	IND	India	YERMARUS TPP	IND0000513	1600.0	16.2949	77.3568	Coal	Oil	NaN	...	WRI	NaN	2018.0	NaN	NaN	0.994875	233.596650	865.400000	Central Electricity Authority	NaN
904	IND	India	Yelesandra Solar Power Plant	WRI1026222	3.0	12.8932	78.1654	Solar	NaN	NaN	...	Industry About	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
905	IND	India	Yelisirur wind power project	WRI1026776	25.5	15.2758	75.5811	Wind	NaN	NaN	...	WRI	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
906	IND	India	ZAWAR MINES	WRI1019901	80.0	24.3500	73.7477	Coal	NaN	NaN	...	WRI	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
907	IND	India	iEnergy Theni Wind Farm	WRI1026761	16.5	9.9344	77.4768	Wind	NaN	NaN	...	WRI	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
908 rows × 25 columns

i have simply loaded the file from Github Repo instead of downloading

Exploratory Data Analysis

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.columns
Index(['country', 'country_long', 'name', 'gppd_idnr', 'capacity_mw',
       'latitude', 'longitude', 'primary_fuel', 'other_fuel1', 'other_fuel2',
       'other_fuel3', 'commissioning_year', 'owner', 'source', 'url',
       'geolocation_source', 'wepp_id', 'year_of_capacity_data',
       'generation_gwh_2013', 'generation_gwh_2014', 'generation_gwh_2015',
       'generation_gwh_2016', 'generation_gwh_2017', 'generation_data_source',
       'estimated_generation_gwh'],
      dtype='object')
df.isnull().sum()
country                       0
country_long                  0
name                          0
gppd_idnr                     0
capacity_mw                   0
latitude                     46
longitude                    46
primary_fuel                  0
other_fuel1                 709
other_fuel2                 907
other_fuel3                 908
commissioning_year          380
owner                       566
source                        0
url                           0
geolocation_source           19
wepp_id                     908
year_of_capacity_data       388
generation_gwh_2013         524
generation_gwh_2014         507
generation_gwh_2015         483
generation_gwh_2016         471
generation_gwh_2017         465
generation_data_source      458
estimated_generation_gwh    908
dtype: int64
df.drop(["other_fuel1", "other_fuel2", "other_fuel3", "owner", "wepp_id", "generation_gwh_2013",
        "generation_gwh_2014", "generation_gwh_2015", "generation_gwh_2016", "generation_gwh_2017",
        "generation_data_source", "estimated_generation_gwh"], axis=1, inplace=True)
df.shape
(908, 13)
df.nunique().to_frame("Unique Values")
Unique Values
country	1
country_long	1
name	908
gppd_idnr	908
capacity_mw	365
latitude	837
longitude	828
primary_fuel	8
commissioning_year	73
source	191
url	304
geolocation_source	3
year_of_capacity_data	1
df.drop(["country", "country_long", "year_of_capacity_data", "name", "gppd_idnr", "url"], axis=1, inplace=True)
print(df.shape)
df.head()
(908, 7)
capacity_mw	latitude	longitude	primary_fuel	commissioning_year	source	geolocation_source
0	2.5	28.1839	73.2407	Solar	2011.0	National Renewable Energy Laboratory	National Renewable Energy Laboratory
1	98.0	24.7663	74.6090	Coal	NaN	Ultratech Cement ltd	WRI
2	39.2	21.9038	69.3732	Wind	NaN	CDM	WRI
3	135.0	23.8712	91.3602	Gas	2004.0	Central Electricity Authority	WRI
4	1800.0	21.9603	82.4091	Coal	2015.0	Central Electricity Authority	WRI
df.isna().sum()
capacity_mw             0
latitude               46
longitude              46
primary_fuel            0
commissioning_year    380
source                  0
geolocation_source     19
dtype: int64
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 908 entries, 0 to 907
Data columns (total 7 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   capacity_mw         908 non-null    float64
 1   latitude            862 non-null    float64
 2   longitude           862 non-null    float64
 3   primary_fuel        908 non-null    object 
 4   commissioning_year  528 non-null    float64
 5   source              908 non-null    object 
 6   geolocation_source  889 non-null    object 
dtypes: float64(4), object(3)
memory usage: 49.8+ KB
# getting list of object data type column names
object_datatype = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        object_datatype.append(x)
print(f"Object Data Type Columns are: ", object_datatype)


# getting the list of float data type column names
float_datatype = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'float64':
        float_datatype.append(x)
print(f"Float Data Type Columns are: ", float_datatype)
Object Data Type Columns are:  ['primary_fuel', 'source', 'geolocation_source']
Float Data Type Columns are:  ['capacity_mw', 'latitude', 'longitude', 'commissioning_year']
# filling missing data for continous values with mean
df["latitude"].fillna(df["latitude"].mean(),inplace=True)
df["longitude"].fillna(df["longitude"].mean(),inplace=True)

# filling missing data for categorical values with mode
df["commissioning_year"].fillna(df["commissioning_year"].mode()[0],inplace=True)
df["geolocation_source"].fillna(df["geolocation_source"].mode()[0],inplace=True)
df.isna().sum()
capacity_mw           0
latitude              0
longitude             0
primary_fuel          0
commissioning_year    0
source                0
geolocation_source    0
dtype: int64
for col in object_datatype:
    print(col)
    print(df[col].value_counts())
    print("="*120)
primary_fuel
Coal       259
Hydro      250
Solar      127
Wind       123
Gas         69
Biomass     50
Oil         21
Nuclear      9
Name: primary_fuel, dtype: int64
========================================================================================================================
source
Central Electricity Authority                                  520
CDM                                                            124
Lancosola                                                       10
National Renewable Energy Laboratory                             8
National Thermal Power Corporation (NTPC)                        6
Rajasthan Renewable Energy Corporation Limited (RREC)            4
Acc Acc ltd                                                      4
Reliance Power Ltd                                               4
Jk Cement ltd                                                    4
Maharashtra State Power Generation Co Ltd (MAHAGENCO)            4
Shri Ssk ltd                                                     3
Karnataka Power Corporation Limited                              3
Ujaas Energy Limited                                             3
SunBorne Energy Gujarat One Private Limited                      3
Hindustan Zinc ltd                                               3
Solairedirect                                                    3
Shree Sugars ltd                                                 3
Ministry of New and Renewable Energy                             3
PEDA                                                             3
Jaypee Ltd jccl)                                                 3
Solar for India                                                  2
Rashtriya & fert                                                 2
Ongc Gas corp                                                    2
Tata Power Solar                                                 2
Shree Cement ltd                                                 2
Ultratech Cement ltd                                             2
Vardham Vardham group                                            2
Adani Power Limite r Er                                          2
Bharat Corp ltd                                                  2
Ambuja Cements ltd                                               2
Jsw Steel ltd                                                    2
Nsl Sugars ltd                                                   2
Lloyds Industries ltd                                            2
Sembcorp                                                         2
Azure Power (Rajasthan) Private Limited                          2
Welspun Solar Punjab Private Limited                             2
Moser Baer Clean Energy Limited (MBCEL)                          2
Ballarpur Industries ltd                                         2
Birla Corp ltd                                                   2
Varam Pvt ltd                                                    1
Precision Technik Private Limited                                1
Manglam Cement ltd                                               1
EMC Limited                                                      1
Datta Ssk ltd                                                    1
Chambal & chem                                                   1
Mukesh Gupta group                                               1
Rk Pvt ltd                                                       1
ACME Cleantech Solutions Private Limited                         1
Topworth Metals ltd                                              1
Purna Ssk ltd                                                    1
Bmm Pvt ltd                                                      1
Gujarat Mineral Development Corporation Limited (GMDC)           1
Dcm Industries ltd                                               1
Essel MP Energy Limited                                          1
Punj Lloyd                                                       1
Sinewave Pvt ltd                                                 1
Unity Power Limite r World                                       1
Videocon Industries Limited                                      1
Urja India edEnewables                                           1
Godawari Energy ltd                                              1
Kesoram Industries ltd                                           1
Hemarus Technologies ltd                                         1
Navbharat                                                        1
Jawahar Ssk ltd                                                  1
Raajratna Energy Holdings Private Limited (REHPL)                1
Gmt Pvt ltd                                                      1
Firestone Trading Private Limited                                1
West Bengal Energy Development Corporation Limited (WBEDCL)      1
Astonfield Management Limited                                    1
Gujarat Industries Power Company Limited (GIPCL)                 1
Kumbhi Ssk ltd                                                   1
Sterling and Wilson                                              1
Pandit Deendayal Petrolium University (PDPU)                     1
Spr Pvt ltd                                                      1
Aftaab Solar Private Limited (ASPL)                              1
Clean Energy edOjects.Com                                        1
Dhariwal Pvt ltd                                                 1
IAEA                                                             1
Orient Cement ltd                                                1
Hiraco India                                                     1
Infratech e M/Pdf                                                1
Jain Solar                                                       1
Orient Power ltd                                                 1
Se Pvt Rsolar                                                    1
Powers Private                                                   1
Lokmangal Lokmangal group                                        1
Maihar Cement ltd                                                1
Sangam Spinners ltd                                              1
Ugar Works ltd                                                   1
Natural Alliend indust                                           1
Tata BP Solar India Limited                                      1
Janki Corp ltd                                                   1
India Ltd icl)                                                   1
Gujarat Urja Vikas Nigam Limited                                 1
Reliance Petrochemicals                                          1
Lingandwi                                                        1
Mukand Mukand ltd                                                1
National Ltd nfl)                                                1
Dr Ssk ltd                                                       1
Manikgarh Manikgarh cement                                       1
Daund Sugar ltd                                                  1
Yashwantrao Krishna ssk                                          1
Sahakar Shiromani vasantrao                                      1
Mula Ssk ltd                                                     1
Technocraft Technocraft group                                    1
Kjs Ahluwalia group                                              1
Bannari Sugars ltd                                               1
Shri Hiranyakeshi ssk                                            1
Hare Pvt ltd                                                     1
P3 Green                                                         1
Amrit Energy Private Limited                                     1
Welspun Solar AP Private Limited                                 1
Manikghar Cement co                                              1
Gm Energy ltd                                                    1
West Coast Paper Mills Ltd.                                      1
Hindustan Pvt lt                                                 1
Rswm Rswm ltd                                                    1
Davangere Co ltd                                                 1
Sri Sugars ltd                                                   1
Est vt Arind                                                     1
Backbone Enterprises Limited                                     1
Greta Energy ltd                                                 1
Sunflag Co ltd                                                   1
Ashok Ssk ltd                                                    1
Mysore Mills ltd                                                 1
Ym Ssk ltd                                                       1
Vikram Vikram cement                                             1
Mangalore & petrochem                                            1
S J Green Park Energy Private Limited                            1
Gem Sugars ltd                                                   1
Coal Mi ICM                                                      1
Purti Sugar ltd                                                  1
Konark Gujarat Private Limited                                   1
Uttam Steels ltd                                                 1
Vasantdada Ssk ltd                                               1
LEPL                                                             1
Indian Oil Corporation (IOC)                                     1
AES Winfra                                                       1
Jai Industries ltd                                               1
Gupta Ltd gepl)                                                  1
Sinarmas Paper ltd                                               1
Sunkon Energy Private Limited                                    1
Solaer                                                           1
Ideal Projects ltd                                               1
ACME Solar Energy                                                1
Madhav Group                                                     1
Mahatma Power ltd                                                1
Shri Vedganga ssk                                                1
Moserbaer Solar                                                  1
Power Private edM/Pdf                                            1
Ambed K Sugar                                                    1
Real Estate e                                                    1
Surana Industries ltd                                            1
Kranti Ssk ltd                                                   1
Meil.In/P                                                        1
Dcm & chem                                                       1
Clover Solar Private Limited (CSPL)                              1
Shri Malaprabha ssk                                              1
Gangakhed Energy ltd                                             1
Tata Co ltd                                                      1
Binani Industries ltd                                            1
Urja Private edEr                                                1
Grace Industries ltd                                             1
Nitin Spinners ltd                                               1
Core Fuels ltd                                                   1
Chettinad Corp ltd                                               1
Mono Steel (India) Ltd                                           1
Nocil Nocil rubber                                               1
Grasim Cement ltd                                                1
Indian Power ltd                                                 1
Godavari Mills ltd                                               1
Indo Synthetics ltd                                              1
Moser Baer Solar Limited (MBSL)                                  1
Rattanindia Power ltd                                            1
Sahakarmaharshi Bhausaheb thor                                   1
Maral Overseas ltd                                               1
Taxus Infrastructure and Power Projects Pvt Ltd                  1
Shamanur Sugars ltd                                              1
Vishwanath Sugars ltd                                            1
Aravali Infrapower Private Limited (AIPL)                        1
Harsha Engineers Limited                                         1
Hira Group                                                       1
Sovox Renewables Private Limited                                 1
S Limited Rsolar                                                 1
National And paper                                               1
Sepset Constructio te                                            1
Bharat Refinery ltd                                              1
SEI Solar Energy Private Limited                                 1
Hothur Pvt ltd                                                   1
Grasim Industries ltd                                            1
CleanEnerg teLeanenergy                                          1
Name: source, dtype: int64
========================================================================================================================
geolocation_source
WRI                                     785
Industry About                          119
National Renewable Energy Laboratory      4
Name: geolocation_source, dtype: int64
========================================================================================================================
Visualization

df.columns
Index(['capacity_mw', 'latitude', 'longitude', 'primary_fuel',
       'commissioning_year', 'source', 'geolocation_source'],
      dtype='object')
try:
    plt.figure(figsize=(10,7))
    col_name = 'primary_fuel'
    values = df[col_name].value_counts()
    index = 0
    ax = sns.countplot(df[col_name], palette="prism")

    for i in ax.get_xticklabels():
        ax.text(index, values[i.get_text()]/2, values[i.get_text()], 
                horizontalalignment="center", fontweight='bold', color='w')
        index += 1
    
    plt.title(f"Count Plot for {col_name}\n")
    plt.ylabel(f"Number of rows")
    plt.show()
    
except Exception as e:
    pass

try:
    plt.figure(figsize=(10,7))
    col_name = 'geolocation_source'
    values = df[col_name].value_counts()
    index = 0
    ax = sns.countplot(df[col_name], palette="prism")

    for i in ax.get_xticklabels():
        ax.text(index, values[i.get_text()]/2, values[i.get_text()], 
                horizontalalignment="center", fontweight='bold', color='w')
        index += 1
    
    plt.title(f"Count Plot for {col_name}\n")
    plt.ylabel(f"Number of rows")
    plt.show()
    
except Exception as e:
    pas

plt.figure(figsize=(15,7))
values = list(df['commissioning_year'].unique())
diag = sns.countplot(df["commissioning_year"], palette="prism")
diag.set_xticklabels(labels=values, rotation=90)
plt.title("Year of power plant operation details\n")
plt.xlabel("List of years weighted by unit-capacity when data is available")
plt.ylabel("Count of Rows in the Dataset")
plt.show()

plt.style.use('ggplot')
sns.scatterplot(x = "commissioning_year", y = "capacity_mw", data = df)
plt.show()

sns.scatterplot(x = "latitude", y = "capacity_mw", data = df)
plt.show()

sns.scatterplot(x = "longitude", y = "capacity_mw", data = df)
plt.show()

plt.style.use('seaborn-pastel')
sns.catplot(x = "primary_fuel", y = "capacity_mw", data = df)
plt.show()

sns.catplot(x = "primary_fuel", y = "latitude", data = df)
plt.show()

sns.catplot(x = "primary_fuel", y = "longitude", data = df)
plt.show()

fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(15,5))
index = 0
ax = ax.flatten()
for col, value in df[float_datatype].items():
    sns.boxplot(y=col, data=df, ax=ax[index], palette="Set1")
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=1.0)
plt.show()

fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(15,5))
index = 0
ax = ax.flatten()
for col, value in df[float_datatype].items():
    sns.distplot(value, ax=ax[index], hist=False, color="g", kde_kws={"shade": True})
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=1.0)
plt.show()

plt.style.use('bmh')
g = sns.pairplot(df)
for ax in g.axes.flat:
    ax.tick_params("x", labelrotation=90)
plt.show()

Encoding all the object datatype columns

# Label Encoder

le = LabelEncoder()
df["primary_fuel"] = le.fit_transform(df["primary_fuel"])
df.head()
capacity_mw	latitude	longitude	primary_fuel	commissioning_year	source	geolocation_source
0	2.5	28.1839	73.2407	6	2011.0	National Renewable Energy Laboratory	National Renewable Energy Laboratory
1	98.0	24.7663	74.6090	1	2013.0	Ultratech Cement ltd	WRI
2	39.2	21.9038	69.3732	7	2013.0	CDM	WRI
3	135.0	23.8712	91.3602	2	2004.0	Central Electricity Authority	WRI
4	1800.0	21.9603	82.4091	1	2015.0	Central Electricity Authority	WRI
# Ordinal Encoder

oe = OrdinalEncoder()
df['geolocation_source'] = oe.fit_transform(df['geolocation_source'].values.reshape(-1,1))
df['source'] = oe.fit_transform(df['source'].values.reshape(-1,1))
df.head()
capacity_mw	latitude	longitude	primary_fuel	commissioning_year	source	geolocation_source
0	2.5	28.1839	73.2407	6	2011.0	109.0	1.0
1	98.0	24.7663	74.6090	1	2013.0	174.0	2.0
2	39.2	21.9038	69.3732	7	2013.0	21.0	2.0
3	135.0	23.8712	91.3602	2	2004.0	22.0	2.0
4	1800.0	21.9603	82.4091	1	2015.0	22.0	2.0
Correlation using a Heatmap

Positive correlation

Negative correlation

upper_triangle = np.triu(df.corr())
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, square=True, fmt='0.3f', 
            annot_kws={'size':10}, cmap="cubehelix", mask=upper_triangle)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

Correlation Bar Plot comparing features with our labels

plt.style.use('dark_background')
df_corr = df.corr()
plt.figure(figsize=(10,5))
df_corr['primary_fuel'].sort_values(ascending=False).drop('primary_fuel').plot.bar()
plt.title("Correlation of Features vs Classification Label\n", fontsize=16)
plt.xlabel("\nFeatures List", fontsize=14)
plt.ylabel("Correlation Value", fontsize=12)
plt.show()

df_corr = df.corr()
plt.figure(figsize=(10,5))
df_corr['capacity_mw'].sort_values(ascending=False).drop('capacity_mw').plot.bar()
plt.title("Correlation of Features vs Regression Label\n", fontsize=16)
plt.xlabel("\nFeatures List", fontsize=14)
plt.ylabel("Correlation Value", fontsize=12)
plt.show()

Using Z Score to remove outliers

z = np.abs(zscore(df))
threshold = 3
df1 = df[(z<3).all(axis = 1)]

print ("Shape of the dataframe before removing outliers: ", df.shape)
print ("Shape of the dataframe after removing outliers: ", df1.shape)
print ("Percentage of data loss post outlier removal: ", (df.shape[0]-df1.shape[0])/df.shape[0]*100)

df=df1.copy() # reassigning the changed dataframe name to our original dataframe name
Shape of the dataframe before removing outliers:  (908, 7)
Shape of the dataframe after removing outliers:  (839, 7)
Percentage of data loss post outlier removal:  7.599118942731277
df.skew()
capacity_mw           1.964097
latitude             -0.109264
longitude             0.846704
primary_fuel          0.419942
commissioning_year   -1.578180
source                1.794155
geolocation_source   -2.114267
dtype: float64
Using Log Transform to fix skewness

for col in float_datatype:
    if df.skew().loc[col]>0.55:
        df[col]=np.log1p(df[col])
Splitting the dataset into 2 variables namely 'X' and 'Y' for feature and classification label

X = df.drop('primary_fuel', axis=1)
Y = df['primary_fuel']
Resolving the class imbalance issue in our label column

Y.value_counts()
1    237
3    219
7    123
6    121
2     64
0     45
5     21
4      9
Name: primary_fuel, dtype: int64
# adding samples to make all the categorical label values same

oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)
Y.value_counts()
7    237
6    237
5    237
4    237
3    237
2    237
1    237
0    237
Name: primary_fuel, dtype: int64
Feature Scaling

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X.head()
capacity_mw	latitude	longitude	commissioning_year	source	geolocation_source
0	-1.755557	1.431962	-0.864147	0.503595	1.375597	-1.211635
1	0.136480	0.829170	-0.463687	0.677660	2.832945	0.365541
2	-0.373700	0.324286	-2.037276	0.677660	-0.597429	0.365541
3	0.316230	0.671293	3.924489	-0.105630	-0.575008	0.365541
4	1.778662	0.334251	1.689214	0.851724	-0.575008	0.365541
Finding best random state for building Classification Models

maxAccu=0
maxRS=0

for i in range(1, 500):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=i)
    lr=LogisticRegression()
    lr.fit(X_train, Y_train)
    pred = lr.predict(X_test)
    acc_score = (accuracy_score(Y_test, pred))*100
    
    if acc_score>maxAccu:
        maxAccu=acc_score
        maxRS=i

print("Best accuracy score is", maxAccu,"on Random State", maxRS)
Best accuracy score is 73.41772151898735 on Random State 375
Machine Learning Model for Classification with Evaluation Metrics

# Classification Model Function

def classify(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=45)
    
    # Training the model
    model.fit(X_train, Y_train)
    
    # Predicting Y_test
    pred = model.predict(X_test)
    
    # Classification Report
    class_report = classification_report(Y_test, pred)
    print("\nClassification Report:\n", class_report)
    
    # Accuracy Score
    acc_score = (accuracy_score(Y_test, pred))*100
    print("Accuracy Score:", acc_score)
    
    # Cross Validation Score
    cv_score = (cross_val_score(model, X, Y, cv=5).mean())*100
    print("Cross Validation Score:", cv_score)
    
    # Result of accuracy minus cv scores
    result = acc_score - cv_score
    print("\nAccuracy Score - Cross Validation Score is", result)
# Logistic Regression

model=LogisticRegression()
classify(model, X, Y)
Classification Report:
               precision    recall  f1-score   support

           0       0.70      0.87      0.78        60
           1       0.56      0.55      0.55        55
           2       0.44      0.27      0.34        44
           3       0.60      0.56      0.58        71
           4       0.70      0.83      0.76        72
           5       0.60      0.42      0.49        60
           6       1.00      0.97      0.98        59
           7       0.79      1.00      0.88        53

    accuracy                           0.69       474
   macro avg       0.67      0.68      0.67       474
weighted avg       0.68      0.69      0.68       474

Accuracy Score: 69.40928270042194
Cross Validation Score: 67.93264824329954

Accuracy Score - Cross Validation Score is 1.476634457122401
# Support Vector Classifier

model=SVC(C=1.0, kernel='rbf', gamma='auto', random_state=42)
classify(model, X, Y)
Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.87      0.90        60
           1       0.65      0.58      0.62        55
           2       0.70      0.32      0.44        44
           3       0.82      0.70      0.76        71
           4       0.73      0.99      0.84        72
           5       0.76      0.83      0.79        60
           6       1.00      0.97      0.98        59
           7       0.78      1.00      0.88        53

    accuracy                           0.80       474
   macro avg       0.80      0.78      0.77       474
weighted avg       0.80      0.80      0.79       474

Accuracy Score: 79.957805907173
Cross Validation Score: 78.6390779058464

Accuracy Score - Cross Validation Score is 1.3187280013265905
# Decision Tree Classifier

model=DecisionTreeClassifier(random_state=21, max_depth=15)
classify(model, X, Y)
Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.97      0.94        60
           1       0.71      0.55      0.62        55
           2       0.70      0.75      0.73        44
           3       0.79      0.76      0.78        71
           4       0.82      0.88      0.85        72
           5       0.89      0.93      0.91        60
           6       0.98      1.00      0.99        59
           7       0.98      1.00      0.99        53

    accuracy                           0.86       474
   macro avg       0.85      0.85      0.85       474
weighted avg       0.85      0.86      0.85       474

Accuracy Score: 85.65400843881856
Cross Validation Score: 86.49868073878629

Accuracy Score - Cross Validation Score is -0.8446722999677263
# Random Forest Classifier

model=RandomForestClassifier(max_depth=15, random_state=111)
classify(model, X, Y)
Classification Report:
               precision    recall  f1-score   support

           0       0.95      1.00      0.98        60
           1       0.78      0.65      0.71        55
           2       0.76      0.77      0.76        44
           3       0.83      0.83      0.83        71
           4       0.87      0.93      0.90        72
           5       0.97      0.95      0.96        60
           6       1.00      1.00      1.00        59
           7       0.98      1.00      0.99        53

    accuracy                           0.90       474
   macro avg       0.89      0.89      0.89       474
weighted avg       0.89      0.90      0.89       474

Accuracy Score: 89.66244725738397
Cross Validation Score: 90.66462991251214

Accuracy Score - Cross Validation Score is -1.0021826551281663
# K Neighbors Classifier

model=KNeighborsClassifier(n_neighbors=15)
classify(model, X, Y)
Classification Report:
               precision    recall  f1-score   support

           0       0.85      0.88      0.87        60
           1       0.67      0.44      0.53        55
           2       0.51      0.43      0.47        44
           3       0.85      0.58      0.69        71
           4       0.78      0.96      0.86        72
           5       0.69      0.85      0.76        60
           6       0.98      1.00      0.99        59
           7       0.71      0.91      0.79        53

    accuracy                           0.77       474
   macro avg       0.76      0.76      0.74       474
weighted avg       0.77      0.77      0.76       474

Accuracy Score: 76.79324894514767
Cross Validation Score: 79.16594917372588

Accuracy Score - Cross Validation Score is -2.3727002285782106
# Extra Trees Classifier

model=ExtraTreesClassifier()
classify(model, X, Y)
Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.98      0.98        60
           1       0.81      0.69      0.75        55
           2       0.77      0.84      0.80        44
           3       0.85      0.82      0.83        71
           4       0.89      0.94      0.92        72
           5       0.97      0.97      0.97        60
           6       1.00      1.00      1.00        59
           7       0.96      1.00      0.98        53

    accuracy                           0.91       474
   macro avg       0.90      0.91      0.90       474
weighted avg       0.91      0.91      0.91       474

Accuracy Score: 90.71729957805907
Cross Validation Score: 91.56131092903763

Accuracy Score - Cross Validation Score is -0.8440113509785618
Hyper parameter tuning on the best Classification ML Model

# Choosing Extra Trees Classifier

fmod_param = {'criterion' : ["gini", "entropy"],
              'n_jobs' : [2, 1, -1],
              'min_samples_split' : [2, 3, 4],
              'max_depth' : [20, 25, 30],
              'random_state' : [42, 45, 111]
             }
GSCV = GridSearchCV(ExtraTreesClassifier(), fmod_param, cv=5)
GSCV.fit(X_train,Y_train)
GridSearchCV(cv=5, estimator=ExtraTreesClassifier(),
             param_grid={'criterion': ['gini', 'entropy'],
                         'max_depth': [20, 25, 30],
                         'min_samples_split': [2, 3, 4], 'n_jobs': [2, 1, -1],
                         'random_state': [42, 45, 111]})
GSCV.best_params_
{'criterion': 'gini',
 'max_depth': 30,
 'min_samples_split': 2,
 'n_jobs': 2,
 'random_state': 45}
Final_Model = ExtraTreesClassifier(criterion="gini", max_depth=30, min_samples_split=4, n_jobs=2, random_state=42)
Classifier = Final_Model.fit(X_train, Y_train)
fmod_pred = Final_Model.predict(X_test)
fmod_acc = (accuracy_score(Y_test, fmod_pred))*100
print("Accuracy score for the Best Model is:", fmod_acc)
Accuracy score for the Best Model is: 90.71729957805907
AUC ROC Curve for multi class label

y_prob = Classifier.predict_proba(X_test)

macro_roc_auc_ovo = roc_auc_score(Y_test, y_prob, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(Y_test, y_prob, multi_class="ovo", average="weighted")
macro_roc_auc_ovr = roc_auc_score(Y_test, y_prob, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(Y_test, y_prob, multi_class="ovr", average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("="*40)
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
One-vs-One ROC AUC scores:
0.992779 (macro),
0.992477 (weighted by prevalence)
========================================
One-vs-Rest ROC AUC scores:
0.992360 (macro),
0.992135 (weighted by prevalence)
Confusion Matrix

class_names = df.columns
metrics.plot_confusion_matrix(Classifier, X_test, Y_test, cmap='mako')
plt.title('\t Confusion Matrix for Extra Trees Classifier \n')
plt.show()

Saving the best Classification ML model

filename = "FinalModel_Classification_GP01.pkl"
joblib.dump(Final_Model, filename)
['FinalModel_Classification_GP01.pkl']
Splitting the dataset into 2 variables namely 'X' and 'Y' for feature and regression label

X = df.drop('capacity_mw', axis=1)
Y = df['capacity_mw']
Feature Scaling

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X.head() # Displaying all the features after applying scaling technique to avoid bias output
latitude	longitude	primary_fuel	commissioning_year	source	geolocation_source
0	1.154079	-0.911736	1.175163	0.468797	1.580000	-1.060641
1	0.598056	-0.567417	-0.983247	0.618009	3.127148	0.400570
2	0.132345	-1.920402	1.606845	0.618009	-0.514600	0.400570
3	0.452429	3.205575	-0.551565	-0.053442	-0.490798	0.400570
4	0.141538	1.283666	-0.983247	0.767220	-0.490798	0.400570
Finding the best random state for building Regression Models

maxAccu=0
maxRS=0

for i in range(1, 1000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=i)
    lr=LinearRegression()
    lr.fit(X_train, Y_train)
    pred = lr.predict(X_test)
    r2 = r2_score(Y_test, pred)
    
    if r2>maxAccu:
        maxAccu=r2
        maxRS=i

print("Best R2 score is", maxAccu,"on Random State", maxRS)
Best R2 score is 0.5383340720045708 on Random State 135
# Regression Model Function

def reg(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=135)
    
    # Training the model
    model.fit(X_train, Y_train)
    
    # Predicting Y_test
    pred = model.predict(X_test)
    
    # RMSE - a lower RMSE score is better than a higher one
    rmse = mean_squared_error(Y_test, pred, squared=False)
    print("RMSE Score is:", rmse)
    
    # R2 score
    r2 = r2_score(Y_test, pred, multioutput='variance_weighted')*100
    print("R2 Score is:", r2)
    
    # Cross Validation Score
    cv_score = (cross_val_score(model, X, Y, cv=5).mean())*100
    print("Cross Validation Score:", cv_score)
    
    # Result of r2 score minus cv score
    result = r2 - cv_score
    print("R2 Score - Cross Validation Score is", result)
# Linear Regression Model

model=LinearRegression()
reg(model, X, Y)
RMSE Score is: 1.2755006251261407
R2 Score is: 53.83340720045708
Cross Validation Score: 42.231770897477126
R2 Score - Cross Validation Score is 11.601636302979955
# Ridge Regression

model=Ridge(alpha=1e-2, normalize=True)
reg(model, X, Y)
RMSE Score is: 1.2754999383789567
R2 Score is: 53.83345691390833
Cross Validation Score: 42.258044119383506
R2 Score - Cross Validation Score is 11.575412794524823
# Lasso Regression

model=Lasso(alpha=1e-2, normalize=True, max_iter=1e5)
reg(model, X, Y)
RMSE Score is: 1.4028822860149064
R2 Score is: 44.15183151646706
Cross Validation Score: 37.52199723471622
R2 Score - Cross Validation Score is 6.6298342817508455
# Support Vector Regression

model=SVR(C=1.0, epsilon=0.2, kernel='poly', gamma='auto')
reg(model, X, Y)
RMSE Score is: 1.1644556142667228
R2 Score is: 61.52201239071935
Cross Validation Score: 47.635622859038264
R2 Score - Cross Validation Score is 13.886389531681083
# Random Forest Regressor

model=RandomForestRegressor(max_depth=2, max_features="sqrt")
reg(model, X, Y)
RMSE Score is: 1.2408141706958205
R2 Score is: 56.310209561440104
Cross Validation Score: 45.931830513916765
R2 Score - Cross Validation Score is 10.378379047523339
# K Neighbors Regressor

KNeighborsRegressor(n_neighbors=2, algorithm='kd_tree')
reg(model, X, Y)
RMSE Score is: 1.2434592207074158
R2 Score is: 56.12374352139048
Cross Validation Score: 45.92465038916207
R2 Score - Cross Validation Score is 10.199093132228413
# Gradient Boosting Regressor

model=GradientBoostingRegressor(loss='quantile', n_estimators=200, max_depth=5)
reg(model, X, Y)
RMSE Score is: 1.59903467556219
R2 Score is: 27.442512525743222
Cross Validation Score: 2.9163511742697157
R2 Score - Cross Validation Score is 24.526161351473505
# Ada Boost Regressor

model=AdaBoostRegressor(n_estimators=300, learning_rate=1.05, random_state=42)
reg(model, X, Y)
RMSE Score is: 1.1131522779037217
R2 Score is: 64.83783310530653
Cross Validation Score: 55.637601015967334
R2 Score - Cross Validation Score is 9.200232089339195
# Extra Trees Regressor

model=ExtraTreesRegressor(n_estimators=200, max_features='sqrt', n_jobs=6)
reg(model, X, Y)
RMSE Score is: 1.0213547174206374
R2 Score is: 70.39809431907575
Cross Validation Score: 63.49722622297922
R2 Score - Cross Validation Score is 6.900868096096531
# XGB Regressor

model=XGBRegressor()
reg(model, X, Y)
RMSE Score is: 1.007419897819593
R2 Score is: 71.20032935535696
Cross Validation Score: 58.83430991273036
R2 Score - Cross Validation Score is 12.366019442626595
Hyper parameter tuning on the best Regression ML Model

# Choosing Extra Trees Regressor

fmod_param = {'criterion' : ['mse', 'mae'],
              'n_estimators' : [100, 200],
              'min_samples_split' : [2, 3],
              'random_state' : [42, 135],
              'n_jobs' : [-1, 1]
             }
GSCV = GridSearchCV(ExtraTreesRegressor(), fmod_param, cv=5)
GSCV.fit(X_train,Y_train)
GridSearchCV(cv=5, estimator=ExtraTreesRegressor(),
             param_grid={'criterion': ['mse', 'mae'],
                         'min_samples_split': [2, 3],
                         'n_estimators': [100, 200], 'n_jobs': [-1, 1],
                         'random_state': [42, 135]})
GSCV.best_params_
{'criterion': 'mse',
 'min_samples_split': 3,
 'n_estimators': 100,
 'n_jobs': -1,
 'random_state': 42}
Final_Model = ExtraTreesRegressor(criterion='mse', min_samples_split=3, n_estimators=100, n_jobs=-1, random_state=42)
Classifier = Final_Model.fit(X_train, Y_train)
fmod_pred = Final_Model.predict(X_test)
fmod_r2 = r2_score(Y_test, fmod_pred)*100
print("R2 score for the Best Model is:", fmod_r2)
R2 score for the Best Model is: 64.67988346483102
Saving the best Regression ML model

filename = "FinalModel_Regression_GP02.pkl"
joblib.dump(Final_Model, filename)
['FinalModel_Regression_GP02.pkl']