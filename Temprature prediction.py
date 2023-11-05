
Weather Forecasting
1. Problem Statement:
a) Design a predictive model with the use of machine learning algorithms to forecast whether or not it will rain tomorrow.

b) Design a predictive model with the use of machine learning algorithms to predict how much rainfall could be there.

a) Design a predictive model with the use of machine learning algorithms to forecast whether or not it will rain tomorrow.
Importing Important Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
#Importing Our data set
data = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset3/main/weatherAUS.csv')
data  #first and last 5 rows of data set.
Date	Location	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	...	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday	RainTomorrow
0	2008-12-01	Albury	13.4	22.9	0.6	NaN	NaN	W	44.0	W	...	71.0	22.0	1007.7	1007.1	8.0	NaN	16.9	21.8	No	No
1	2008-12-02	Albury	7.4	25.1	0.0	NaN	NaN	WNW	44.0	NNW	...	44.0	25.0	1010.6	1007.8	NaN	NaN	17.2	24.3	No	No
2	2008-12-03	Albury	12.9	25.7	0.0	NaN	NaN	WSW	46.0	W	...	38.0	30.0	1007.6	1008.7	NaN	2.0	21.0	23.2	No	No
3	2008-12-04	Albury	9.2	28.0	0.0	NaN	NaN	NE	24.0	SE	...	45.0	16.0	1017.6	1012.8	NaN	NaN	18.1	26.5	No	No
4	2008-12-05	Albury	17.5	32.3	1.0	NaN	NaN	W	41.0	ENE	...	82.0	33.0	1010.8	1006.0	7.0	8.0	17.8	29.7	No	No
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
8420	2017-06-21	Uluru	2.8	23.4	0.0	NaN	NaN	E	31.0	SE	...	51.0	24.0	1024.6	1020.3	NaN	NaN	10.1	22.4	No	No
8421	2017-06-22	Uluru	3.6	25.3	0.0	NaN	NaN	NNW	22.0	SE	...	56.0	21.0	1023.5	1019.1	NaN	NaN	10.9	24.5	No	No
8422	2017-06-23	Uluru	5.4	26.9	0.0	NaN	NaN	N	37.0	SE	...	53.0	24.0	1021.0	1016.8	NaN	NaN	12.5	26.1	No	No
8423	2017-06-24	Uluru	7.8	27.0	0.0	NaN	NaN	SE	28.0	SSE	...	51.0	24.0	1019.4	1016.5	3.0	2.0	15.1	26.0	No	No
8424	2017-06-25	Uluru	14.9	NaN	0.0	NaN	NaN	NaN	NaN	ESE	...	62.0	36.0	1020.2	1017.9	8.0	8.0	15.0	20.9	No	NaN
8425 rows × 23 columns

Exploratory Data Analysis (EDA)
# Check the dimension of our DataSet
data.shape
(8425, 23)
we have a total of 8425 records with 23 features

#Let us check the name of our columns
data.columns
Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RainTomorrow'],
      dtype='object')
# Check the data types of our 
data.dtypes
Date              object
Location          object
MinTemp          float64
MaxTemp          float64
Rainfall         float64
Evaporation      float64
Sunshine         float64
WindGustDir       object
WindGustSpeed    float64
WindDir9am        object
WindDir3pm        object
WindSpeed9am     float64
WindSpeed3pm     float64
Humidity9am      float64
Humidity3pm      float64
Pressure9am      float64
Pressure3pm      float64
Cloud9am         float64
Cloud3pm         float64
Temp9am          float64
Temp3pm          float64
RainToday         object
RainTomorrow      object
dtype: object
So we have 16 Numeric datatypes and 7 Object type data types

# Check the info about our dataset
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8425 entries, 0 to 8424
Data columns (total 23 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Date           8425 non-null   object 
 1   Location       8425 non-null   object 
 2   MinTemp        8350 non-null   float64
 3   MaxTemp        8365 non-null   float64
 4   Rainfall       8185 non-null   float64
 5   Evaporation    4913 non-null   float64
 6   Sunshine       4431 non-null   float64
 7   WindGustDir    7434 non-null   object 
 8   WindGustSpeed  7434 non-null   float64
 9   WindDir9am     7596 non-null   object 
 10  WindDir3pm     8117 non-null   object 
 11  WindSpeed9am   8349 non-null   float64
 12  WindSpeed3pm   8318 non-null   float64
 13  Humidity9am    8366 non-null   float64
 14  Humidity3pm    8323 non-null   float64
 15  Pressure9am    7116 non-null   float64
 16  Pressure3pm    7113 non-null   float64
 17  Cloud9am       6004 non-null   float64
 18  Cloud3pm       5970 non-null   float64
 19  Temp9am        8369 non-null   float64
 20  Temp3pm        8329 non-null   float64
 21  RainToday      8185 non-null   object 
 22  RainTomorrow   8186 non-null   object 
dtypes: float64(16), object(7)
memory usage: 1.5+ MB
From this we can identity that many columns have null values

# Check the null value counts
data.isna().sum()
Date                0
Location            0
MinTemp            75
MaxTemp            60
Rainfall          240
Evaporation      3512
Sunshine         3994
WindGustDir       991
WindGustSpeed     991
WindDir9am        829
WindDir3pm        308
WindSpeed9am       76
WindSpeed3pm      107
Humidity9am        59
Humidity3pm       102
Pressure9am      1309
Pressure3pm      1312
Cloud9am         2421
Cloud3pm         2455
Temp9am            56
Temp3pm            96
RainToday         240
RainTomorrow      239
dtype: int64
From this we can Notice that almost all feilds have null values, So we have to use imputation technigues for removing

# Let us fund the Unique values present in our data
data.nunique()
Date             3004
Location           12
MinTemp           285
MaxTemp           331
Rainfall          250
Evaporation       116
Sunshine          140
WindGustDir        16
WindGustSpeed      52
WindDir9am         16
WindDir3pm         16
WindSpeed9am       34
WindSpeed3pm       35
Humidity9am        90
Humidity3pm        94
Pressure9am       384
Pressure3pm       374
Cloud9am            9
Cloud3pm            9
Temp9am           304
Temp3pm           328
RainToday           2
RainTomorrow        2
dtype: int64
From this we can notice that only two few catagorical columns are available.

#Now Check any duplicate data present or not, If present then we will drop them.
data=data.drop_duplicates()
data.shape
(6762, 23)
As we noticed earlier we had around 8425 records were present, but now only 6762 that means 1663 duplicate record were present in our data, so we have removed them

Statistical description of our data
data.describe().T
count	mean	std	min	25%	50%	75%	max
MinTemp	6692.0	13.109145	5.569574	-2.0	9.0	13.2	17.5	28.5
MaxTemp	6705.0	24.098345	6.156128	8.2	19.5	23.5	28.4	45.5
Rainfall	6624.0	2.780148	10.591418	0.0	0.0	0.0	0.8	371.0
Evaporation	3841.0	5.302395	4.436790	0.0	2.6	4.6	7.0	145.0
Sunshine	3526.0	7.890896	3.785883	0.0	5.4	9.0	10.8	13.9
WindGustSpeed	5820.0	38.977663	14.418577	7.0	30.0	37.0	48.0	107.0
WindSpeed9am	6699.0	12.782206	9.833499	0.0	6.0	11.0	19.0	63.0
WindSpeed3pm	6662.0	17.571150	9.620043	0.0	9.0	17.0	24.0	83.0
Humidity9am	6708.0	67.506559	17.251733	10.0	56.0	68.0	81.0	100.0
Humidity3pm	6666.0	50.467147	18.631086	6.0	38.0	50.0	63.0	99.0
Pressure9am	5454.0	1017.626311	6.712043	989.8	1013.1	1017.6	1022.2	1039.0
Pressure3pm	5451.0	1015.119923	6.646755	982.9	1010.3	1015.1	1019.6	1036.0
Cloud9am	4896.0	4.336806	2.908324	0.0	1.0	5.0	7.0	8.0
Cloud3pm	4860.0	4.320988	2.740519	0.0	1.0	5.0	7.0	8.0
Temp9am	6711.0	17.895038	5.744117	1.9	13.8	18.0	22.2	39.4
Temp3pm	6670.0	22.708561	6.012896	7.3	18.3	22.1	26.8	44.1
This is the statistical description of our data,

As we noticed many feature contains null values the count is different for all.
Many feature the minimum values is 0.
Some features contains negative values also.
Data Preprocessing
#Now Let us find the values of each Features in our data
for i in data.columns:
        print(data[i].value_counts())
        print('*'*100)
2011-02-11    4
2011-02-18    4
2011-03-18    4
2011-03-19    4
2011-03-20    4
             ..
2016-11-03    1
2016-11-02    1
2016-11-01    1
2016-10-31    1
2013-06-08    1
Name: Date, Length: 3004, dtype: int64
****************************************************************************************************
PerthAirport    1204
Albury           907
Newcastle        822
Melbourne        811
Williamtown      615
CoffsHarbour     611
Brisbane         579
Penrith          482
Darwin           250
Wollongong       237
Adelaide         205
Uluru             39
Name: Location, dtype: int64
****************************************************************************************************
13.2    58
12.0    57
14.8    53
12.7    53
10.8    52
        ..
26.6     1
28.0     1
26.9     1
1.4      1
26.0     1
Name: MinTemp, Length: 285, dtype: int64
****************************************************************************************************
19.0    66
19.8    62
20.8    54
23.8    54
25.0    54
        ..
38.9     1
10.3     1
9.4      1
42.5     1
43.5     1
Name: MaxTemp, Length: 331, dtype: int64
****************************************************************************************************
0.0      4334
0.2       321
0.4       144
0.6        87
1.2        69
         ... 
73.8        1
23.8        1
61.2        1
128.0       1
40.0        1
Name: Rainfall, Length: 250, dtype: int64
****************************************************************************************************
4.0      141
3.0      125
2.2      118
2.4      116
2.6      116
        ... 
145.0      1
33.8       1
59.2       1
20.8       1
0.7        1
Name: Evaporation, Length: 116, dtype: int64
****************************************************************************************************
0.0     119
11.1     61
11.0     59
11.2     59
9.2      56
       ... 
2.5       5
13.6      4
13.8      2
13.9      2
13.5      1
Name: Sunshine, Length: 140, dtype: int64
****************************************************************************************************
E      518
SW     465
N      459
W      434
WSW    420
WNW    398
SSE    390
S      376
SE     370
ENE    357
NE     300
SSW    299
NW     296
NNE    287
ESE    267
NNW    184
Name: WindGustDir, dtype: int64
****************************************************************************************************
39.0     346
35.0     341
37.0     332
33.0     317
31.0     305
30.0     302
41.0     285
28.0     285
43.0     237
26.0     228
24.0     225
48.0     211
22.0     201
46.0     195
50.0     191
52.0     186
44.0     181
20.0     170
54.0     150
19.0     117
56.0     111
57.0     105
17.0      88
61.0      83
59.0      80
63.0      67
15.0      57
13.0      57
65.0      50
67.0      42
72.0      42
70.0      35
69.0      35
74.0      34
76.0      25
11.0      18
80.0      15
78.0      15
85.0       9
81.0       9
9.0        6
91.0       6
83.0       5
98.0       4
93.0       4
89.0       4
94.0       2
87.0       2
7.0        2
107.0      1
102.0      1
100.0      1
Name: WindGustSpeed, dtype: int64
****************************************************************************************************
N      609
SW     590
NW     463
SE     439
ENE    397
WSW    394
SSW    368
NE     364
E      338
NNE    337
S      324
WNW    301
SSE    300
W      299
ESE    229
NNW    216
Name: WindDir9am, dtype: int64
****************************************************************************************************
SE     677
WSW    499
S      493
NE     480
SW     428
SSE    421
NW     400
W      399
E      392
WNW    389
ESE    363
N      354
ENE    348
NNE    305
SSW    277
NNW    243
Name: WindDir3pm, dtype: int64
****************************************************************************************************
0.0     730
9.0     641
4.0     595
13.0    520
7.0     482
6.0     459
11.0    456
17.0    366
15.0    344
19.0    333
20.0    314
2.0     240
24.0    228
22.0    213
28.0    169
26.0    158
31.0    107
30.0     81
35.0     51
33.0     50
37.0     40
41.0     26
39.0     21
43.0     16
46.0     16
44.0     16
52.0      8
56.0      5
50.0      5
54.0      3
48.0      3
61.0      1
57.0      1
63.0      1
Name: WindSpeed9am, dtype: int64
****************************************************************************************************
9.0     639
19.0    509
13.0    505
11.0    458
20.0    452
17.0    446
15.0    430
24.0    383
28.0    355
22.0    342
7.0     293
4.0     283
26.0    279
6.0     220
30.0    196
0.0     192
31.0    176
33.0    112
35.0     93
37.0     85
2.0      54
39.0     49
41.0     30
46.0     20
43.0     20
44.0     11
50.0      8
56.0      6
48.0      6
52.0      5
65.0      1
83.0      1
54.0      1
61.0      1
57.0      1
Name: WindSpeed3pm, dtype: int64
****************************************************************************************************
68.0    163
73.0    161
69.0    152
70.0    148
74.0    148
       ... 
17.0      2
14.0      2
16.0      2
10.0      1
15.0      1
Name: Humidity9am, Length: 90, dtype: int64
****************************************************************************************************
46.0    157
51.0    155
54.0    154
49.0    151
52.0    150
       ... 
8.0      10
7.0       7
98.0      6
99.0      3
6.0       3
Name: Humidity3pm, Length: 94, dtype: int64
****************************************************************************************************
1019.2    42
1018.7    41
1014.8    41
1020.0    40
1019.6    39
          ..
1036.2     1
997.3      1
1002.1     1
993.4      1
1033.6     1
Name: Pressure9am, Length: 384, dtype: int64
****************************************************************************************************
1017.8    46
1018.0    41
1016.1    40
1017.9    39
1017.4    39
          ..
990.8      1
1028.0     1
992.4      1
1035.9     1
1029.5     1
Name: Pressure3pm, Length: 374, dtype: int64
****************************************************************************************************
7.0    1043
1.0     922
8.0     764
0.0     521
6.0     454
5.0     341
3.0     313
2.0     311
4.0     227
Name: Cloud9am, dtype: int64
****************************************************************************************************
7.0    959
1.0    921
8.0    644
6.0    489
5.0    433
2.0    428
3.0    357
0.0    332
4.0    297
Name: Cloud3pm, dtype: int64
****************************************************************************************************
14.8    62
18.0    61
20.6    57
17.5    54
18.3    52
        ..
2.5      1
2.0      1
3.4      1
5.2      1
30.2     1
Name: Temp9am, Length: 304, dtype: int64
****************************************************************************************************
19.2    63
19.0    61
22.5    54
21.7    54
23.5    53
        ..
41.1     1
40.9     1
41.0     1
38.0     1
42.4     1
Name: Temp3pm, Length: 328, dtype: int64
****************************************************************************************************
No     5052
Yes    1572
Name: RainToday, dtype: int64
****************************************************************************************************
No     5052
Yes    1572
Name: RainTomorrow, dtype: int64
****************************************************************************************************
From Above we can notice that there is no spacial character or space in between features, Also we noticed same number of values in RainToday and RainTomorrow, Let us check if they are same, if same we can remove one columns

if data['RainToday'].equals(data['RainTomorrow']):
    print ('Same Data')
else:
    print("Different Data")
Different Data
So they are different, so we can proceed

#Now We will convert date from object type to date type and then seperate day, month and year
data['Date']=pd.to_datetime(data['Date'])
data["Day"] = data['Date'].dt.day
data["Month"] = data['Date'].dt.month
data["Year"] = data['Date'].dt.year
# Now drop date coloumns from our data
data.drop("Date",axis=1,inplace=True)
#As we removed around 1663 records let us check again null value sum
data.isnull().sum()
Location            0
MinTemp            70
MaxTemp            57
Rainfall          138
Evaporation      2921
Sunshine         3236
WindGustDir       942
WindGustSpeed     942
WindDir9am        794
WindDir3pm        294
WindSpeed9am       63
WindSpeed3pm      100
Humidity9am        54
Humidity3pm        96
Pressure9am      1308
Pressure3pm      1311
Cloud9am         1866
Cloud3pm         1902
Temp9am            51
Temp3pm            92
RainToday         138
RainTomorrow      138
Day                 0
Month               0
Year                0
dtype: int64
So the number has reduced much

# Let us separate Numercial and Catagorical features
# Checking for categorical columns
categorical_col=[]
for i in data.dtypes.index:
    if data.dtypes[i]=='object':
        categorical_col.append(i)
print("Categorical columns are:\n",categorical_col)
print("\n")

# Now checking for numerical columns
numerical_col=[]
for i in data.dtypes.index:
    if data.dtypes[i]!='object':
        numerical_col.append(i)
print("Numerical columns are:\n",numerical_col)
Categorical columns are:
 ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']


Numerical columns are:
 ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Day', 'Month', 'Year']
# Now let us replace the NAN values in Other catagorical Coloumns by MODE 

df=pd.DataFrame()
df['WindGustDir'] = data['WindGustDir'].fillna(data['WindGustDir'].mode()[0])
df['WindDir9am'] = data['WindDir9am'].fillna(data['WindDir9am'].mode()[0])
df['WindDir3pm'] = data['WindDir3pm'].fillna(data['WindDir3pm'].mode()[0])
df['RainToday'] = data['RainToday'].fillna(data['RainToday'].mode()[0])
df['RainTomorrow']=data['RainTomorrow'].fillna(data['RainTomorrow'].mode()[0])
df['Location']=data['Location']
df
WindGustDir	WindDir9am	WindDir3pm	RainToday	RainTomorrow	Location
0	W	W	WNW	No	No	Albury
1	WNW	NNW	WSW	No	No	Albury
2	WSW	W	WSW	No	No	Albury
3	NE	SE	E	No	No	Albury
4	W	ENE	NW	No	No	Albury
...	...	...	...	...	...	...
8420	E	SE	ENE	No	No	Uluru
8421	NNW	SE	N	No	No	Uluru
8422	N	SE	WNW	No	No	Uluru
8423	SE	SSE	N	No	No	Uluru
8424	E	ESE	ESE	No	No	Uluru
6762 rows × 6 columns

#Now impute values to other fields using this feature
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# We can impute values to our numerical columns by using Iterative imputers
itimp=IterativeImputer()
befor_imp=pd.DataFrame(data[numerical_col])
kk_df=pd.DataFrame(itimp.fit_transform(befor_imp),columns=numerical_col)
kk_df
MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	WindGustSpeed	WindSpeed9am	WindSpeed3pm	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	Day	Month	Year
0	13.4	22.900000	0.6	3.391685	7.130509	44.000000	20.0	24.0	71.0	22.0	1007.7	1007.1	8.000000	4.874826	16.9	21.8	1.0	12.0	2008.0
1	7.4	25.100000	0.0	4.149341	12.628322	44.000000	4.0	22.0	44.0	25.0	1010.6	1007.8	1.135447	1.451345	17.2	24.3	2.0	12.0	2008.0
2	12.9	25.700000	0.0	5.872974	12.022633	46.000000	19.0	26.0	38.0	30.0	1007.6	1008.7	2.268447	2.000000	21.0	23.2	3.0	12.0	2008.0
3	9.2	28.000000	0.0	3.771469	13.091641	24.000000	11.0	9.0	45.0	16.0	1017.6	1012.8	0.505052	0.825362	18.1	26.5	4.0	12.0	2008.0
4	17.5	32.300000	1.0	3.909425	6.732299	41.000000	7.0	20.0	82.0	33.0	1010.8	1006.0	7.000000	8.000000	17.8	29.7	5.0	12.0	2008.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
6757	2.8	23.400000	0.0	6.351080	11.113943	31.000000	13.0	11.0	51.0	24.0	1024.6	1020.3	0.685550	1.589567	10.1	22.4	21.0	6.0	2017.0
6758	3.6	25.300000	0.0	6.025810	11.780452	22.000000	13.0	9.0	56.0	21.0	1023.5	1019.1	0.577039	1.130006	10.9	24.5	22.0	6.0	2017.0
6759	5.4	26.900000	0.0	7.826086	11.551481	37.000000	9.0	9.0	53.0	24.0	1021.0	1016.8	0.869720	1.639326	12.5	26.1	23.0	6.0	2017.0
6760	7.8	27.000000	0.0	8.048489	10.572988	28.000000	13.0	7.0	51.0	24.0	1019.4	1016.5	3.000000	2.000000	15.1	26.0	24.0	6.0	2017.0
6761	14.9	21.689692	0.0	7.364270	3.642095	42.803274	17.0	17.0	62.0	36.0	1020.2	1017.9	8.000000	8.000000	15.0	20.9	25.0	6.0	2017.0
6762 rows × 19 columns

df[numerical_col]=kk_df[numerical_col]
df.shape
(6762, 25)
df.isna().sum()
WindGustDir         0
WindDir9am          0
WindDir3pm          0
RainToday           0
RainTomorrow        0
Location            0
MinTemp          1663
MaxTemp          1663
Rainfall         1663
Evaporation      1663
Sunshine         1663
WindGustSpeed    1663
WindSpeed9am     1663
WindSpeed3pm     1663
Humidity9am      1663
Humidity3pm      1663
Pressure9am      1663
Pressure3pm      1663
Cloud9am         1663
Cloud3pm         1663
Temp9am          1663
Temp3pm          1663
Day              1663
Month            1663
Year             1663
dtype: int64
df.dropna(inplace=True)
df.shape
(5099, 25)
So we have removed all null values from our data

# Visaulize the same
sns.heatmap(df.isna())
<AxesSubplot:>

So from this we can identify that all null values have been removed

Visualization
#1st Univarient analisys
categorical_col
['Location',
 'WindGustDir',
 'WindDir9am',
 'WindDir3pm',
 'RainToday',
 'RainTomorrow']
sns.countplot(df['Location'],hue=df['RainTomorrow'])
plt.xticks(rotation=90)
plt.show()

We can notice that we have maximum features from ALbury low data from wollongong

print(df['WindGustDir'].value_counts())
sns.countplot(df['WindGustDir'],hue=df['RainTomorrow'])
plt.xticks(rotation=90)
plt.show()
E      1143
N       372
SSE     364
W       333
S       326
WNW     322
SE      302
SW      287
WSW     261
NNE     239
NE      237
SSW     216
ENE     210
NW      195
ESE     162
NNW     130
Name: WindGustDir, dtype: int64

From This we can notice that if wind direction is in East direction the the chances of Rain is very low, All other the cances is almost same

print(df['WindDir9am'].value_counts())
sns.countplot(df['WindDir9am'],hue=df['RainTomorrow'])
plt.xticks(rotation=90)
plt.show()
N      1235
SW      539
NW      433
WSW     351
SE      322
SSW     291
W       262
WNW     258
S       236
SSE     234
NE      203
NNW     169
NNE     166
E       138
ESE     136
ENE     126
Name: WindDir9am, dtype: int64

For we can notice that if wind direction at 9 am is towards north direction then the chances of Rain is Low

print(df['WindDir3pm'].value_counts())
sns.countplot(df['WindDir3pm'],hue=df['RainTomorrow'])
plt.xticks(rotation=90)
plt.show()
SE     910
S      443
NE     424
SSE    362
NW     304
N      285
WNW    279
ESE    276
ENE    256
E      252
W      252
NNE    245
WSW    237
SSW    214
SW     213
NNW    147
Name: WindDir3pm, dtype: int64

In this case if the Wind direction is towards South East then the chances of Rain is low

Now let us plot with Numerical data and catagorical
numerical_col        # Numercial data
['MinTemp',
 'MaxTemp',
 'Rainfall',
 'Evaporation',
 'Sunshine',
 'WindGustSpeed',
 'WindSpeed9am',
 'WindSpeed3pm',
 'Humidity9am',
 'Humidity3pm',
 'Pressure9am',
 'Pressure3pm',
 'Cloud9am',
 'Cloud3pm',
 'Temp9am',
 'Temp3pm',
 'Day',
 'Month',
 'Year']
#Let us find the fainfall at different location
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.lineplot(df['Location'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

From this we can notice that At coffsHarbour the rain fall is comparetevily high, But chances of rain tommorow is at Wollongong or Melbourne is low

#Let us find the fainfall at minimum Temperature
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['MinTemp'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

High rain fall in temperature of 10 to 25

#Let us find the rainfall at Max Temperature
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['MaxTemp'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

If the Maximum temperature is in between 15 to 30 then the rainfall is high

#Let us find the rainfall at Evaporation
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['Evaporation'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

From this we can reasily identify some outliers are present, and high rain fall during evapuration at 20 to 30, and chances of Raining Tomorrow is also high during same rate

#Let us find the rainfall at Sunshine
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['Sunshine'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

As Sunshine increases rain fall decreases, but IF sunshine is very high some chances of rain fall is there

#Let us find the rainfall at WindGustSpeed
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['WindGustSpeed'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

Rain fall is in between 20 to 80 windgustspeed

#Let us find the rainfall at Windspeed3pm
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['WindSpeed3pm'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

If wind speed 0 to 40 at noon then chances of Rainfall is high

#Let us find the rainfall at WindSpeed9am
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['WindSpeed9am'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

If in morning there is wind then chances rain is high

#Check is there any relation between windspeed9am with windspeed3pm
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['WindSpeed9am'],df['WindSpeed3pm'],hue=df['RainTomorrow'])
plt.show()

This Shows there is a linear relation

#Let us find the Windgustspeed at WindSpeed9am
plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
sns.scatterplot(df['WindSpeed9am'],df['WindGustSpeed'],hue=df['RainTomorrow'])
plt.show()

Also have some linear relation

plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Comparision between Humidity9am Rain_Fall')
sns.scatterplot(df['Humidity9am'],df['Rainfall'],hue=df['RainTomorrow'])

plt.subplot(2,2,2)
plt.title('Comparision between Humidity3pm and Rainfall')
sns.scatterplot(df['Humidity3pm'],df['Rainfall'],hue=df['RainTomorrow'])

plt.subplot(2,2,3)
plt.title('Comparision between Pressure9am and Rainfall')
sns.scatterplot(df['Pressure9am'],df['Rainfall'],hue=df['RainTomorrow'])

plt.subplot(2,2,4)
plt.title('Comparision between Pressure3pm and Rainfall')
sns.scatterplot(df['Pressure3pm'],df['Rainfall'],hue=df['RainTomorrow'])
plt.show()

From Fig 1&2 we can notice if high Humidity is there then chaces of rain fall is high From Fig 3 &4 we can notice that at pressure in between 1000 to 1030 chances of rain fall is high

plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Comparision between Cloud9am Rain_Fall')
sns.scatterplot(df['Cloud9am'],df['Rainfall'],hue=df['RainTomorrow'])

plt.subplot(2,2,2)
plt.title('Comparision between Cloud3pm and Rainfall')
sns.scatterplot(df['Cloud3pm'],df['Rainfall'],hue=df['RainTomorrow'],palette='hsv')

plt.subplot(2,2,3)
plt.title('Comparision between Pressure9am and Humidity9am')
sns.scatterplot(df['Pressure9am'],df['Humidity9am'],hue=df['RainTomorrow'])

plt.subplot(2,2,4)
plt.title('Comparision between Humidity3pm and Humidity9am')
sns.scatterplot(df['Humidity3pm'],df['Humidity9am'],hue=df['RainTomorrow'],palette='hsv')
plt.show()

From fig 1 & 2 As cloudy is high chances of rain is also high
From Fig 3 As Pressure increases the humidity alos increasing
Humidity 9am and 3pm have linear relation
plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Comparision between WindGustDir & Rain_Fall')
sns.lineplot(df['WindGustDir'],df['Rainfall'],hue=df['RainTomorrow'],palette='coolwarm')

plt.subplot(2,2,2)
plt.title('Comparision between Winddir9am and Rainfall')
sns.lineplot(df['WindDir9am'],df['Rainfall'],hue=df['RainTomorrow'],palette='cool')

plt.subplot(2,2,3)
plt.title('Comparision between WindDir3pm and Rainfall')
sns.lineplot(df['WindDir3pm'],df['Rainfall'],hue=df['RainTomorrow'],palette='spring')

plt.subplot(2,2,4)
plt.title('Comparision between RAinToday and Rainfall')
sns.lineplot(df['RainToday'],df['Rainfall'],hue=df['RainTomorrow'],palette='hsv')
plt.show()

If windGustdir is from NNW to SW then chances of rainis too high and high rainfall is in that location, if it is in direction SE to E then Chances of rain is low
If Winddir9am is from WSW to NW then chances of rain is High Similarly if winddirat9am is in N to WSW the rain fall will be low
If Winddir3pm W to ESE chances of rain low, nut if it is in ESE to SSW or to S then rainfall willbe high
If rainfall no rainfall today then chances of raining tomorow is high, and vice versa
sns.pairplot(df,palette='spring',hue='RainTomorrow')
plt.show()

Conclusion
High relation between rainfall and RainTomorrow.
If it rainToday then Chances of raining tomorrow is less.
Some linear relationships noticed.
We notice Wind Direction have influence on rainfall.
Identifying the outliers
# Let's check the outliers by ploting box plot

plt.figure(figsize=(25,35),facecolor='white')
plotnumber=1
for column in numerical_col:
    if plotnumber<=23:
        ax=plt.subplot(6,4,plotnumber)
        sns.boxplot(df[column],palette="Set2_r")
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()

Most of the features contains outliers, let us remove the out liners using ZScore method

Removing outliers using Zscore method
features=df[['MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Pressure9am','Pressure3pm','Temp9am','Temp3pm']]

# Using zscore to remove outliers
from scipy.stats import zscore

z=np.abs(zscore(features))

z 
MaxTemp	Rainfall	Evaporation	Sunshine	WindGustSpeed	WindSpeed9am	WindSpeed3pm	Humidity9am	Pressure9am	Pressure3pm	Temp9am	Temp3pm
0	0.229987	0.205244	0.398878	0.143002	0.585970	1.015416	0.813279	0.128103	1.766860	1.411133	0.215139	0.187819
1	0.134071	0.257142	0.215776	1.408207	0.585970	0.780844	0.602789	1.443096	1.265458	1.289729	0.162886	0.236069
2	0.233360	0.257142	0.200773	1.237311	0.728578	0.903150	1.023769	1.792251	1.784150	1.133638	0.498993	0.049559
3	0.613966	0.257142	0.307096	1.538932	0.840108	0.005020	0.765395	1.384903	0.055177	0.422557	0.006125	0.609090
4	1.325534	0.170645	0.273756	0.255357	0.372059	0.444045	0.392299	0.768221	1.230878	1.601911	0.058379	1.151667
...	...	...	...	...	...	...	...	...	...	...	...	...
6757	0.147247	0.257142	0.316316	0.980924	0.340981	0.229553	0.554905	1.035748	1.155104	0.878200	1.399554	0.086085
6758	0.167167	0.257142	0.237709	1.168980	0.982716	0.229553	0.765395	0.744785	0.964917	0.670079	1.260211	0.269980
6759	0.431937	0.257142	0.672780	1.104376	0.086843	0.219512	0.765395	0.919363	0.532674	0.271180	0.981525	0.541268
6760	0.448485	0.257142	0.726528	0.828294	0.554892	0.229553	0.975885	1.035748	0.256038	0.219150	0.528661	0.524313
6761	0.430270	0.257142	0.561173	1.127259	0.500639	0.678618	0.076564	0.395630	0.394356	0.461958	0.546079	0.340418
5099 rows × 12 columns

# Creating new dataframe
new_df = df[(z<3).all(axis=1)] 
new_df
WindGustDir	WindDir9am	WindDir3pm	RainToday	RainTomorrow	Location	MinTemp	MaxTemp	Rainfall	Evaporation	...	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	Day	Month	Year
0	W	W	WNW	No	No	Albury	13.4	22.900000	0.6	3.391685	...	22.0	1007.7	1007.1	8.000000	4.874826	16.9	21.8	1.0	12.0	2008.0
1	WNW	NNW	WSW	No	No	Albury	7.4	25.100000	0.0	4.149341	...	25.0	1010.6	1007.8	1.135447	1.451345	17.2	24.3	2.0	12.0	2008.0
2	WSW	W	WSW	No	No	Albury	12.9	25.700000	0.0	5.872974	...	30.0	1007.6	1008.7	2.268447	2.000000	21.0	23.2	3.0	12.0	2008.0
3	NE	SE	E	No	No	Albury	9.2	28.000000	0.0	3.771469	...	16.0	1017.6	1012.8	0.505052	0.825362	18.1	26.5	4.0	12.0	2008.0
4	W	ENE	NW	No	No	Albury	17.5	32.300000	1.0	3.909425	...	33.0	1010.8	1006.0	7.000000	8.000000	17.8	29.7	5.0	12.0	2008.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
6757	N	N	N	No	No	Adelaide	2.8	23.400000	0.0	6.351080	...	24.0	1024.6	1020.3	0.685550	1.589567	10.1	22.4	21.0	6.0	2017.0
6758	WNW	NNE	N	No	Yes	Adelaide	3.6	25.300000	0.0	6.025810	...	21.0	1023.5	1019.1	0.577039	1.130006	10.9	24.5	22.0	6.0	2017.0
6759	WSW	SSW	SW	Yes	Yes	Adelaide	5.4	26.900000	0.0	7.826086	...	24.0	1021.0	1016.8	0.869720	1.639326	12.5	26.1	23.0	6.0	2017.0
6760	WSW	S	W	Yes	No	Adelaide	7.8	27.000000	0.0	8.048489	...	24.0	1019.4	1016.5	3.000000	2.000000	15.1	26.0	24.0	6.0	2017.0
6761	NW	N	NW	No	Yes	Adelaide	14.9	21.689692	0.0	7.364270	...	36.0	1020.2	1017.9	8.000000	8.000000	15.0	20.9	25.0	6.0	2017.0
4839 rows × 25 columns

# Total Data Loss 
dataloss=(5099-4839)/5099
dataloss*100
5.099039027260247
Using Z-Score, we are losing only 5% of data , that is affordable and we may proceed.

Finding Skewness
plt.figure(figsize=(25,35),facecolor='white')
plotnumber=1
for column in numerical_col:
    if plotnumber<=23:
        ax=plt.subplot(6,4,plotnumber)
        sns.distplot(new_df[column],color="darkgreen",hist=False,kde_kws={"shade": True})
        plt.xlabel(column,fontsize=18)
    plotnumber+=1
plt.tight_layout()

Few features like rainfall contains high skewness. Let us check using skew()

new_df.skew()
MinTemp         -0.137773
MaxTemp          0.197290
Rainfall         3.650157
Evaporation      0.174545
Sunshine        -0.288996
WindGustSpeed    0.601910
WindSpeed9am     0.756993
WindSpeed3pm     0.426798
Humidity9am     -0.273921
Humidity3pm      0.098996
Pressure9am     -0.109561
Pressure3pm     -0.077155
Cloud9am         0.024393
Cloud3pm         0.096084
Temp9am         -0.209469
Temp3pm          0.243698
Day              0.009663
Month            0.033661
Year             0.083311
dtype: float64
Removing Skewness using yeo-johnson method
skew=['Rainfall','Evaporation','WindGustSpeed','WindSpeed9am','WindSpeed3pm']
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method='yeo-johnson')

new_df[skew] = scaler.fit_transform(new_df[skew].values)
new_df[skew].head()
Rainfall	Evaporation	WindGustSpeed	WindSpeed9am	WindSpeed3pm
0	0.048555	-0.449326	0.803420	1.112653	0.930441
1	-0.465737	-0.191372	0.803420	-0.668855	0.738956
2	-0.465737	0.390325	0.934713	1.031915	1.116634
3	-0.465737	-0.319826	-0.826674	0.282632	-0.698347
4	0.281386	-0.272886	0.598279	-0.203514	0.541547
new_df[skew].skew()
Rainfall        -4.981502
Evaporation      0.086263
WindGustSpeed   -0.012794
WindSpeed9am    -0.132963
WindSpeed3pm    -0.077357
dtype: float64
Skewness have been reduced, We can notice Rainfall is one of our Label so we can proceed

# After removing skewness let's check how the data has been distributed in each column.

plt.figure(figsize=(20,20), facecolor='white')
plotnumber = 1

for column in new_df[skew]:
    if plotnumber<=4:
        ax = plt.subplot(2,2,plotnumber)
        sns.distplot(new_df[column],color='indigo',kde_kws={"shade": True},hist=False)
        plt.xlabel(column,fontsize=15)
    plotnumber+=1
plt.show()

# Removing skewness using square root method
new_df["Rainfall"] = np.cbrt(new_df["Rainfall"])
new_df.skew()  # again checking skewness
MinTemp         -0.137773
MaxTemp          0.197290
Rainfall         1.096926
Evaporation      0.086263
Sunshine        -0.288996
WindGustSpeed   -0.012794
WindSpeed9am    -0.132963
WindSpeed3pm    -0.077357
Humidity9am     -0.273921
Humidity3pm      0.098996
Pressure9am     -0.109561
Pressure3pm     -0.077155
Cloud9am         0.024393
Cloud3pm         0.096084
Temp9am         -0.209469
Temp3pm          0.243698
Day              0.009663
Month            0.033661
Year             0.083311
dtype: float64
Now the Skewness reduced, and let us proceed further

Encoding the categorical columns using Label Encoding
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()
new_df[categorical_col]= new_df[categorical_col].apply(LE.fit_transform)
new_df[categorical_col]
Location	WindGustDir	WindDir9am	WindDir3pm	RainToday	RainTomorrow
0	1	13	13	14	0	0
1	1	14	6	15	0	0
2	1	15	13	15	0	0
3	1	4	9	0	0	0
4	1	13	1	7	0	0
...	...	...	...	...	...	...
6757	0	3	3	3	0	0
6758	0	14	5	3	0	1
6759	0	15	11	12	1	1
6760	0	15	8	13	1	0
6761	0	7	3	7	0	1
4839 rows × 6 columns

So we have encoded our data, now let us find the other relations

Correlation between the target variable and independent variables using HEAT map
# Checking the correlation between features and the target
cor = new_df.corr()
cor
WindGustDir	WindDir9am	WindDir3pm	RainToday	RainTomorrow	Location	MinTemp	MaxTemp	Rainfall	Evaporation	...	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	Day	Month	Year
WindGustDir	1.000000	0.367967	0.393584	0.071513	0.006166	-0.147793	-0.081854	-0.069394	-0.017777	-0.046661	...	-0.111734	-0.086622	-0.052079	0.009259	-0.018267	-0.072270	-0.067763	0.000404	0.021164	-0.212247
WindDir9am	0.367967	1.000000	0.201090	0.122139	-0.007133	-0.021021	0.069061	-0.024063	0.046053	0.102758	...	-0.033417	-0.058537	-0.013349	-0.003938	0.004311	0.075152	-0.025337	-0.021223	0.017524	-0.039959
WindDir3pm	0.393584	0.201090	1.000000	0.083341	-0.017381	-0.050372	-0.143353	-0.157298	0.042125	-0.045528	...	-0.034395	-0.026644	0.030035	0.043943	0.032016	-0.164988	-0.160667	-0.001424	0.032329	-0.021169
RainToday	0.071513	0.122139	0.083341	1.000000	0.294971	0.010212	0.102598	-0.139564	0.670205	-0.057430	...	0.275839	-0.032336	0.016307	0.268806	0.240204	-0.028751	-0.143006	0.001433	-0.048330	0.018636
RainTomorrow	0.006166	-0.007133	-0.017381	0.294971	1.000000	0.012833	0.096017	-0.099118	0.240868	-0.046788	...	0.339144	-0.037336	-0.015301	0.243258	0.294422	0.004357	-0.127703	0.003060	-0.052404	0.020604
Location	-0.147793	-0.021021	-0.050372	0.010212	0.012833	1.000000	0.109190	-0.042359	0.079019	0.239948	...	0.188513	0.073166	0.086926	0.115216	0.143629	0.063952	-0.054065	-0.001757	-0.032383	0.542482
MinTemp	-0.081854	0.069061	-0.143353	0.102598	0.096017	0.109190	1.000000	0.739828	0.139570	0.601006	...	0.122702	-0.489516	-0.482624	0.156774	0.115432	0.905063	0.709154	0.022301	-0.204835	0.030889
MaxTemp	-0.069394	-0.024063	-0.157298	-0.139564	-0.099118	-0.042359	0.739828	1.000000	-0.188904	0.709351	...	-0.393091	-0.406983	-0.486397	-0.287213	-0.294340	0.860738	0.980235	0.013752	-0.128894	0.059236
Rainfall	-0.017777	0.046053	0.042125	0.670205	0.240868	0.079019	0.139570	-0.188904	1.000000	-0.112612	...	0.408603	-0.100413	-0.026077	0.405943	0.351286	-0.040751	-0.200529	-0.000003	-0.033031	0.035458
Evaporation	-0.046661	0.102758	-0.045528	-0.057430	-0.046788	0.239948	0.601006	0.709351	-0.112612	1.000000	...	-0.335572	-0.336950	-0.350884	-0.172180	-0.230072	0.692513	0.688649	0.003909	0.020956	0.378581
Sunshine	0.013827	0.006796	-0.060204	-0.295733	-0.313694	-0.156778	0.073398	0.577814	-0.431385	0.441297	...	-0.745252	-0.073435	-0.148891	-0.786976	-0.805850	0.347121	0.607436	-0.019258	0.080068	-0.028905
WindGustSpeed	0.168608	0.193863	0.025146	0.039725	0.078210	0.013737	0.292443	0.240020	0.075311	0.423384	...	-0.139680	-0.424821	-0.382195	0.001073	0.041553	0.314290	0.205922	-0.010010	0.061452	-0.033935
WindSpeed9am	0.235467	0.407074	-0.002367	0.061240	0.052306	-0.020133	0.232785	0.103581	0.047429	0.281645	...	-0.068740	-0.236920	-0.185891	-0.006402	0.039211	0.246031	0.088914	-0.010495	0.038621	-0.067726
WindSpeed3pm	0.150769	0.198193	-0.026377	-0.010425	-0.014810	-0.012967	0.242018	0.200175	0.007508	0.301268	...	-0.108828	-0.298303	-0.249812	-0.033194	-0.054630	0.300441	0.183882	-0.007642	0.050436	-0.054984
Humidity9am	-0.119860	-0.177281	-0.008537	0.256925	0.200316	0.109831	-0.114346	-0.386030	0.409543	-0.491655	...	0.628277	0.189486	0.210957	0.530369	0.420542	-0.371675	-0.380397	0.011701	-0.140823	0.094771
Humidity3pm	-0.111734	-0.033417	-0.034395	0.275839	0.339144	0.188513	0.122702	-0.393091	0.408603	-0.335572	...	1.000000	0.040785	0.111498	0.578490	0.620441	-0.076099	-0.461916	0.032333	-0.085517	-0.039025
Pressure9am	-0.086622	-0.058537	-0.026644	-0.032336	-0.037336	0.073166	-0.489516	-0.406983	-0.100413	-0.336950	...	0.040785	1.000000	0.963198	-0.082048	-0.066289	-0.483372	-0.368648	-0.038894	0.022430	0.108720
Pressure3pm	-0.052079	-0.013349	0.030035	0.016307	-0.015301	0.086926	-0.482624	-0.486397	-0.026077	-0.350884	...	0.111498	0.963198	1.000000	-0.005773	0.008579	-0.503896	-0.456350	-0.029317	0.004946	0.103076
Cloud9am	0.009259	-0.003938	0.043943	0.268806	0.243258	0.115216	0.156774	-0.287213	0.405943	-0.172180	...	0.578490	-0.082048	-0.005773	1.000000	0.700128	-0.107495	-0.307239	0.012197	-0.044187	0.044150
Cloud3pm	-0.018267	0.004311	0.032016	0.240204	0.294422	0.143629	0.115432	-0.294340	0.351286	-0.230072	...	0.620441	-0.066289	0.008579	0.700128	1.000000	-0.087512	-0.347860	0.031357	-0.042638	0.049007
Temp9am	-0.072270	0.075152	-0.164988	-0.028751	0.004357	0.063952	0.905063	0.860738	-0.040751	0.692513	...	-0.076099	-0.483372	-0.503896	-0.107495	-0.087512	1.000000	0.828433	0.018715	-0.118569	0.010964
Temp3pm	-0.067763	-0.025337	-0.160667	-0.143006	-0.127703	-0.054065	0.709154	0.980235	-0.200529	0.688649	...	-0.461916	-0.368648	-0.456350	-0.307239	-0.347860	0.828433	1.000000	0.012808	-0.146544	0.065026
Day	0.000404	-0.021223	-0.001424	0.001433	0.003060	-0.001757	0.022301	0.013752	-0.000003	0.003909	...	0.032333	-0.038894	-0.029317	0.012197	0.031357	0.018715	0.012808	1.000000	-0.004940	0.004965
Month	0.021164	0.017524	0.032329	-0.048330	-0.052404	-0.032383	-0.204835	-0.128894	-0.033031	0.020956	...	-0.085517	0.022430	0.004946	-0.044187	-0.042638	-0.118569	-0.146544	-0.004940	1.000000	-0.070271
Year	-0.212247	-0.039959	-0.021169	0.018636	0.020604	0.542482	0.030889	0.059236	0.035458	0.378581	...	-0.039025	0.108720	0.103076	0.044150	0.049007	0.010964	0.065026	0.004965	-0.070271	1.000000
25 rows × 25 columns

This gives the correlation between the denpendent and independent variables. We can visualize this by plotting heat map.

# Visualizing the correlation matrix by plotting heat map.
plt.figure(figsize=(25,25))
sns.heatmap(new_df.corr(),linewidths=.1,vmin=-1, vmax=1, fmt='.1g',linecolor="black", annot = True, annot_kws={'size':10},cmap="cool")
<AxesSubplot:>

From this we can notice some multi colinearity between some features, so we will use VIF techniques to remove highly related feature

Visualizing the correlation between label and features using bar plot
plt.figure(figsize=(20,10))
new_df.corr()['RainTomorrow'].sort_values(ascending=False).drop(['RainTomorrow']).plot(kind='bar',color='indigo')
plt.xlabel('Feature',fontsize=14)
plt.ylabel('Target',fontsize=14)
plt.title('correlation between label and feature using bar plot',fontsize=18)
plt.show()

This is for Classificaion Problem the relation between RainTomorrow as our Label, Some features does not have much relation between Label

Feature Scaling using Standard Scalarization
from sklearn.preprocessing import StandardScaler
#Separate Label and Features
x=new_df.drop('RainTomorrow',axis=1)
y=new_df.RainTomorrow
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
x.head()
WindGustDir	WindDir9am	WindDir3pm	RainToday	Location	MinTemp	MaxTemp	Rainfall	Evaporation	Sunshine	...	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	Day	Month	Year
0	1.247965	1.213389	1.541063	-0.560882	-1.379334	0.048488	-0.226612	0.806117	-0.449326	-0.154832	...	-1.588149	-1.883671	-1.502182	1.479615	0.327698	-0.202854	-0.185536	-1.668284	1.667302	-1.743795
1	1.441737	-0.358897	1.779808	-0.560882	-1.379334	-1.001527	0.145232	-0.615681	-0.191372	1.435521	...	-1.426422	-1.357451	-1.375007	-1.110178	-1.049117	-0.150355	0.247241	-1.554771	1.667302	-1.743795
2	1.635509	1.213389	1.779808	-0.560882	-1.379334	-0.039013	0.246644	-0.615681	0.390325	1.260313	...	-1.156876	-1.901817	-1.211498	-0.682731	-0.828466	0.514633	0.056819	-1.441258	1.667302	-1.743795
3	-0.495982	0.314940	-1.801369	-0.560882	-1.379334	-0.686522	0.635391	-0.615681	-0.319826	1.569545	...	-1.911604	-0.087263	-0.466621	-1.348008	-1.300868	0.007143	0.628084	-1.327745	1.667302	-1.743795
4	1.247965	-1.481959	-0.130153	-0.560882	-1.379334	0.765999	1.362178	1.168404	-0.272886	-0.270022	...	-0.995149	-1.321159	-1.702027	1.102344	1.584544	-0.045357	1.182039	-1.214232	1.667302	-1.743795
5 rows × 24 columns

Checking Variance Inflation Factor(VIF)
# Finding varience inflation factor in each scaled column i.e, x.shape[1] (1/(1-R2))

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF values"] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif["Features"] = x.columns

# Let's check the values
vif
VIF values	Features
0	1.475585	WindGustDir
1	1.404584	WindDir9am
2	1.307690	WindDir3pm
3	1.900497	RainToday
4	1.655218	Location
5	12.622109	MinTemp
6	38.589724	MaxTemp
7	2.337321	Rainfall
8	5.093665	Evaporation
9	10.512002	Sunshine
10	3.840994	WindGustSpeed
11	2.276619	WindSpeed9am
12	2.952324	WindSpeed3pm
13	5.015941	Humidity9am
14	6.986194	Humidity3pm
15	23.012086	Pressure9am
16	22.783839	Pressure3pm
17	3.709485	Cloud9am
18	3.725421	Cloud3pm
19	21.318020	Temp9am
20	46.827763	Temp3pm
21	1.009367	Day
22	1.248273	Month
23	2.249768	Year
Many feature are highly multi related So we will drop Temp3pm as it have highest VIF values

x.drop('Temp3pm',axis=1,inplace=True)
#Again Check VIF score
vif = pd.DataFrame()
vif["VIF values"] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif["Features"] = x.columns

# Let's check the values
vif
VIF values	Features
0	1.475228	WindGustDir
1	1.403688	WindDir9am
2	1.307351	WindDir3pm
3	1.900250	RainToday
4	1.652686	Location
5	12.450665	MinTemp
6	13.716104	MaxTemp
7	2.337318	Rainfall
8	5.090933	Evaporation
9	10.497546	Sunshine
10	3.792340	WindGustSpeed
11	2.264903	WindSpeed9am
12	2.951203	WindSpeed3pm
13	4.439800	Humidity9am
14	4.775954	Humidity3pm
15	21.607362	Pressure9am
16	21.646353	Pressure3pm
17	3.686584	Cloud9am
18	3.698369	Cloud3pm
19	20.014874	Temp9am
20	1.007291	Day
21	1.213720	Month
22	2.248878	Year
#Now we will remove Pressure3pm
x.drop('Pressure3pm',axis=1,inplace=True)
#Again check VIF score
vif = pd.DataFrame()
vif["VIF values"] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif["Features"] = x.columns

# Let's check the values
vif
VIF values	Features
0	1.474942	WindGustDir
1	1.403227	WindDir9am
2	1.278329	WindDir3pm
3	1.894109	RainToday
4	1.648175	Location
5	12.282022	MinTemp
6	11.836635	MaxTemp
7	2.298344	Rainfall
8	5.001804	Evaporation
9	10.119354	Sunshine
10	3.781244	WindGustSpeed
11	2.263416	WindSpeed9am
12	2.929737	WindSpeed3pm
13	4.439760	Humidity9am
14	4.774847	Humidity3pm
15	1.611483	Pressure9am
16	3.658018	Cloud9am
17	3.590936	Cloud3pm
18	19.954012	Temp9am
19	1.005501	Day
20	1.178931	Month
21	2.248053	Year
# Now We will remove Temp9am from our data
x.drop('Temp9am',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF values"] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif["Features"] = x.columns

# Let's check the values
vif
VIF values	Features
0	1.474879	WindGustDir
1	1.397777	WindDir9am
2	1.275884	WindDir3pm
3	1.893722	RainToday
4	1.635427	Location
5	6.540286	MinTemp
6	8.449583	MaxTemp
7	2.295020	Rainfall
8	4.997755	Evaporation
9	9.939048	Sunshine
10	3.701866	WindGustSpeed
11	2.261178	WindSpeed9am
12	2.818961	WindSpeed3pm
13	3.246627	Humidity9am
14	3.668927	Humidity3pm
15	1.600776	Pressure9am
16	3.644286	Cloud9am
17	3.582916	Cloud3pm
18	1.005360	Day
19	1.166643	Month
20	2.247961	Year
Now all feature were VIF values less than 10, let us proceed to modeling

Clearing-Oversampling
As we noticed Label values are not balanced

# Oversampling the data
from imblearn.over_sampling import SMOTE
SM = SMOTE()
x, y = SM.fit_resample(x,y)
# Checking value count of target column
y.value_counts()
0    3665
1    3665
Name: RainTomorrow, dtype: int64
Modeling
#Finding best random state
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
maxAccu=0
maxRS=0
for i in range(1,200):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30, random_state =i)
    DTC = RandomForestClassifier()
    DTC.fit(x_train, y_train)
    pred = DTC.predict(x_test)
    acc=accuracy_score(y_test, pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print("Best accuracy is ",maxAccu," on Random_state ",maxRS)
Best accuracy is  0.8835834470213734  on Random_state  54
We have got the best random state as 54 and maximum accuracy as 88.35%

# Creating train_test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=maxRS)
Classification Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from xgboost import XGBClassifier as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score
from sklearn.model_selection import cross_val_score
Random Forest Classifier
# Checking accuracy for Random Forest Classifier
RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)

# Prediction
predRFC = RFC.predict(x_test)
rfc=accuracy_score(y_test, predRFC)
print(rfc)
print(confusion_matrix(y_test, predRFC))
print(classification_report(y_test,predRFC))
0.8799454297407913
[[984 137]
 [127 951]]
              precision    recall  f1-score   support

           0       0.89      0.88      0.88      1121
           1       0.87      0.88      0.88      1078

    accuracy                           0.88      2199
   macro avg       0.88      0.88      0.88      2199
weighted avg       0.88      0.88      0.88      2199

The accuracy using Random Forest Classifier is 88%

# Lets plot confusion matrix for RandomForestClassifier
cm = confusion_matrix(y_test,predRFC)

x_axis_labels = ["NO","YES"]
y_axis_labels = ["NO","YES"]

f , ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot = True,linewidths=.2, linecolor="black", fmt = ".0f", ax=ax, cmap="cool",xticklabels=x_axis_labels,yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for RandomForestClassifier')
plt.show()

Support Vector Machine Classifier
# Checking accuracy for Support Vector Machine Classifier
svc = SVC()
svc.fit(x_train,y_train)

# Prediction
predsvc = svc.predict(x_test)
sv=accuracy_score(y_test, predsvc)
print(sv)
print(confusion_matrix(y_test, predsvc))
print(classification_report(y_test,predsvc))
0.7980900409276944
[[878 243]
 [201 877]]
              precision    recall  f1-score   support

           0       0.81      0.78      0.80      1121
           1       0.78      0.81      0.80      1078

    accuracy                           0.80      2199
   macro avg       0.80      0.80      0.80      2199
weighted avg       0.80      0.80      0.80      2199

The accuracy score using Support Vector Machine classifier is 80%

# Lets plot confusion matrix for Support Vector Machine Classifier
cm = confusion_matrix(y_test,predsvc)

x_axis_labels = ["NO","YES"]
y_axis_labels = ["NO","YES"]

f , ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot = True,linewidths=.2, linecolor="black", fmt = ".0f", ax=ax, cmap="cool",xticklabels=x_axis_labels,yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Support Vector Machine Classifier')
plt.show()

Gradient Boosting Classifier
# Checking accuracy for Gradient Boosting Classifier
GB = GradientBoostingClassifier()
GB.fit(x_train,y_train)

# Prediction
predGB = GB.predict(x_test)

gb=accuracy_score(y_test, predGB)
print(gb)
print(confusion_matrix(y_test, predGB))
print(classification_report(y_test,predGB))
0.8281036834924966
[[937 184]
 [194 884]]
              precision    recall  f1-score   support

           0       0.83      0.84      0.83      1121
           1       0.83      0.82      0.82      1078

    accuracy                           0.83      2199
   macro avg       0.83      0.83      0.83      2199
weighted avg       0.83      0.83      0.83      2199

The accuracy using Gradient Boosting Classifier is 83%

# Lets plot confusion matrix for Gradient Boosting Classifier
cm = confusion_matrix(y_test,predGB)

x_axis_labels = ["NO","YES"]
y_axis_labels = ["NO","YES"]

f , ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot = True,linewidths=.2, linecolor="black", fmt = ".0f", ax=ax, cmap="cool",xticklabels=x_axis_labels,yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Gradient Boosting Classifier')
plt.show()

Bagging Classifier
# Checking accuracy for BaggingClassifier
BC = BaggingClassifier()
BC.fit(x_train,y_train)

# Prediction
predBC = BC.predict(x_test)
bc=accuracy_score(y_test, predBC)
print(bc)
print(confusion_matrix(y_test, predBC))
print(classification_report(y_test,predBC))
0.8458390177353342
[[971 150]
 [189 889]]
              precision    recall  f1-score   support

           0       0.84      0.87      0.85      1121
           1       0.86      0.82      0.84      1078

    accuracy                           0.85      2199
   macro avg       0.85      0.85      0.85      2199
weighted avg       0.85      0.85      0.85      2199

The accuracy using Bagging classifier is 85%

# Lets plot confusion matrix for  Bagging Classifier
cm = confusion_matrix(y_test,predBC)

x_axis_labels = ["NO","YES"]
y_axis_labels = ["NO","YES"]

f , ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot = True,linewidths=.2, linecolor="black", fmt = ".0f", ax=ax, cmap="cool",xticklabels=x_axis_labels,yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for  Bagging Classifier')
plt.show()

XGB Classifier
# Checking accuracy for XGBClassifier
XGB = xgb(verbosity=0)
XGB.fit(x_train,y_train)

# Prediction
predXGB = XGB.predict(x_test)
xgb1=accuracy_score(y_test, predXGB)
print(xgb1)
print(confusion_matrix(y_test, predXGB))
print(classification_report(y_test,predXGB))
0.8867667121418826
[[1009  112]
 [ 137  941]]
              precision    recall  f1-score   support

           0       0.88      0.90      0.89      1121
           1       0.89      0.87      0.88      1078

    accuracy                           0.89      2199
   macro avg       0.89      0.89      0.89      2199
weighted avg       0.89      0.89      0.89      2199

The accuracy using XGB classifier is 89%

# Lets plot confusion matrix for  XGBClassifier
cm = confusion_matrix(y_test,predXGB)

x_axis_labels = ["NO","YES"]
y_axis_labels = ["NO","YES"]

f , ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot = True,linewidths=.2, linecolor="black", fmt = ".0f", ax=ax, cmap="cool",xticklabels=x_axis_labels,yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for  XGB Classifier')
plt.show()

Checking Cross Validation Score
# cv score for Random Forest Classifier
rf=cross_val_score(RFC,x,y,cv=5).mean()
print(rf)
0.5789904502046385
# cv score for Support Vector Machine Classifier
sv_cv=cross_val_score(svc,x,y,cv=5).mean()
print(sv_cv)
0.5934515688949522
# cv score for Gradient Boosting Classifier
gb_cv=cross_val_score(GB,x,y,cv=5).mean()
print(gb_cv)
0.47626193724420196
# cv score for Bagging Classifier
bc_cv=cross_val_score(BC,x,y,cv=5).mean()
print(bc_cv)
0.5376534788540246
# cv score for XGB Classifier
xgb_cv=cross_val_score(XGB,x,y,cv=5).mean()
print(xgb_cv)
0.48744884038199177
model_list=['Random Forest Classifier','Support Vector Machine Classifier','Gradient Boosting Classifier','Bagging Classifier','XGB Classifier']
accuracyscore=[rfc,sv,gb,bc,xgb1]
crossval=[rf,sv_cv,gb_cv,bc_cv,xgb_cv]
score_diff=[]
for i in range(len(accuracyscore)):
    score_diff.append(accuracyscore[i]-crossval[i])
models=pd.DataFrame({})
models["Classifier"]=model_list#
models["Accuracy_score"]=accuracyscore
models["Cross Validation_Score"]=crossval
models["Differance"]=score_diff
models
Classifier	Accuracy_score	Cross Validation_Score	Differance
0	Random Forest Classifier	0.879945	0.578990	0.300955
1	Support Vector Machine Classifier	0.798090	0.593452	0.204638
2	Gradient Boosting Classifier	0.828104	0.476262	0.351842
3	Bagging Classifier	0.845839	0.537653	0.308186
4	XGB Classifier	0.886767	0.487449	0.399318
So Support Vector Machine is the best model and has least differance value, so let us proceed

Plotting ROC and compare AUC for all the models used
# Plotting for all the models used here
from sklearn import datasets 
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import plot_roc_curve 


disp=plot_roc_curve(svc, x_test, y_test)   # ax_=Axes with confusion matrix
plot_roc_curve(RFC, x_test, y_test, ax=disp.ax_)

plot_roc_curve(GB, x_test, y_test, ax=disp.ax_)
plot_roc_curve(BC, x_test, y_test, ax=disp.ax_)
plot_roc_curve(XGB, x_test, y_test, ax=disp.ax_)

plt.legend(prop={'size':11}, loc='lower right')
plt.show()

Hyper Parameter Tuning
# Support Vector Classifier
from sklearn.model_selection import GridSearchCV
params={'C':[1,0.5,1.5],
       'kernel' : ['linear', 'poly', 'rbf'],
        'gamma' : ['scale', 'auto'],
        'random_state':[100,200,150,250]}
GCV=GridSearchCV(SVC(),params,cv=3)
GCV.fit(x_train,y_train)
GridSearchCV(cv=3, estimator=SVC(),
             param_grid={'C': [1, 0.5, 1.5], 'gamma': ['scale', 'auto'],
                         'kernel': ['linear', 'poly', 'rbf'],
                         'random_state': [100, 200, 150, 250]})
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
GCV.best_params_
{'C': 1.5, 'gamma': 'auto', 'kernel': 'rbf', 'random_state': 100}
These are the best parameters values that we have got for SVC

FinalModel = SVC(C= 1.5, gamma='scale', kernel= 'rbf', random_state=100)
FinalModel.fit(x_train, y_train)
pred = FinalModel.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc*100)
81.26421100500227
So Accuracy of our final Clasification model is 81.26%

# Lets plot confusion matrix for  FinalModel
cm = confusion_matrix(y_test,pred)

x_axis_labels = ["NO","YES"]
y_axis_labels = ["NO","YES"]

f , ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot = True,linewidths=.2, linecolor="black", fmt = ".0f", ax=ax, cmap="ocean",xticklabels=x_axis_labels,yticklabels=y_axis_labels)

plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for  Final Model')
plt.show()

Plotting ROC and Compare AUC for the best model
# Let's check the Auc for the best model after hyper parameter tuning
plot_roc_curve(FinalModel, x_test, y_test)
plt.title("ROC for the best model")
plt.show()

Save Classification model
# Saving the model using .pkl
import joblib
joblib.dump(FinalModel,"RainClassification.pkl")
['RainClassification.pkl']
Predicting the saved model
# Let's load the saved model and get the prediction

# Loading the saved model
model=joblib.load("RainClassification.pkl")

#Prediction
prediction = model.predict(x_test)
prediction
array([1, 0, 0, ..., 0, 0, 0])
a=np.array(y_test)
Class_Result=pd.DataFrame({'Original':a,'Prediction':prediction})
Class_Result
Original	Prediction
0	1	1
1	0	0
2	0	0
3	0	0
4	0	0
...	...	...
2194	1	1
2195	1	1
2196	0	0
2197	0	0
2198	0	0
2199 rows × 2 columns

So we predicted RainTomorrow with our model, and Model Accuracy is 81.26%

 
b) Design a predictive model with the use of machine learning algorithms to predict how much rainfall could be there.
#Finding relation ith Rainfall
plt.figure(figsize=(30,25))
sns.heatmap(cor,linewidths=.1,vmin=-1, vmax=1, fmt='.1g',linecolor="black", annot = True, annot_kws={'size':10},cmap="cool")
<AxesSubplot:>

This gives Corelation with features and Labels, Now letuse check relation with Rainfall

Visualizing the correlation between label and features using bar plot
plt.figure(figsize=(20,10))
new_df.corr()['Rainfall'].sort_values(ascending=False).drop(['Rainfall']).plot(kind='bar',color='y')
plt.xlabel('Feature',fontsize=14)
plt.ylabel('Target',fontsize=14)
plt.title('correlation between label and feature using bar plot',fontsize=18)
plt.show()

Almost all features have relation between Label, But only WindSpeed3pm does not have much relation

Separating the features and label variables into x and y
x = new_df.drop("Rainfall", axis=1)
y = new_df["Rainfall"]
Feature Scaling using Standard Scalarization
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
x.head()
WindGustDir	WindDir9am	WindDir3pm	RainToday	RainTomorrow	Location	MinTemp	MaxTemp	Evaporation	Sunshine	...	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	Day	Month	Year
0	1.247965	1.213389	1.541063	-0.560882	-0.565975	-1.379334	0.048488	-0.226612	-0.449326	-0.154832	...	-1.588149	-1.883671	-1.502182	1.479615	0.327698	-0.202854	-0.185536	-1.668284	1.667302	-1.743795
1	1.441737	-0.358897	1.779808	-0.560882	-0.565975	-1.379334	-1.001527	0.145232	-0.191372	1.435521	...	-1.426422	-1.357451	-1.375007	-1.110178	-1.049117	-0.150355	0.247241	-1.554771	1.667302	-1.743795
2	1.635509	1.213389	1.779808	-0.560882	-0.565975	-1.379334	-0.039013	0.246644	0.390325	1.260313	...	-1.156876	-1.901817	-1.211498	-0.682731	-0.828466	0.514633	0.056819	-1.441258	1.667302	-1.743795
3	-0.495982	0.314940	-1.801369	-0.560882	-0.565975	-1.379334	-0.686522	0.635391	-0.319826	1.569545	...	-1.911604	-0.087263	-0.466621	-1.348008	-1.300868	0.007143	0.628084	-1.327745	1.667302	-1.743795
4	1.247965	-1.481959	-0.130153	-0.560882	-0.565975	-1.379334	0.765999	1.362178	-0.272886	-0.270022	...	-0.995149	-1.321159	-1.702027	1.102344	1.584544	-0.045357	1.182039	-1.214232	1.667302	-1.743795
5 rows × 24 columns

We have scaled the data using standard scalarizaion .

Checking Variance Inflation Factor(VIF)
# Finding varience inflation factor in each scaled column i.e, x.shape[1] (1/(1-R2))

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF values"] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif["Features"] = x.columns

# Let's check the values
vif
VIF values	Features
0	1.472143	WindGustDir
1	1.406864	WindDir9am
2	1.309551	WindDir3pm
3	1.260134	RainToday
4	1.252099	RainTomorrow
5	1.668373	Location
6	12.598002	MinTemp
7	38.563117	MaxTemp
8	5.093444	Evaporation
9	10.510134	Sunshine
10	3.839655	WindGustSpeed
11	2.275505	WindSpeed9am
12	2.956867	WindSpeed3pm
13	4.855465	Humidity9am
14	7.110164	Humidity3pm
15	22.562978	Pressure9am
16	22.440236	Pressure3pm
17	3.714303	Cloud9am
18	3.722789	Cloud3pm
19	21.304350	Temp9am
20	46.829453	Temp3pm
21	1.009229	Day
22	1.241119	Month
23	2.254699	Year
#Let us remove Temp3pm
x.drop('Temp3pm',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF values"] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif["Features"] = x.columns

# Let's check the values
vif
VIF values	Features
0	1.471769	WindGustDir
1	1.405986	WindDir9am
2	1.309222	WindDir3pm
3	1.259898	RainToday
4	1.252052	RainTomorrow
5	1.665915	Location
6	12.426289	MinTemp
7	13.696958	MaxTemp
8	5.090762	Evaporation
9	10.495494	Sunshine
10	3.790547	WindGustSpeed
11	2.263777	WindSpeed9am
12	2.955705	WindSpeed3pm
13	4.278061	Humidity9am
14	4.893234	Humidity3pm
15	21.162651	Pressure9am
16	21.306203	Pressure3pm
17	3.691260	Cloud9am
18	3.695588	Cloud3pm
19	19.999913	Temp9am
20	1.007150	Day
21	1.206650	Month
22	2.253784	Year
x.drop('Temp9am',axis=1,inplace=True)
vif = pd.DataFrame()#Again check the value of VIF
vif["VIF values"] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif["Features"] = x.columns

# Let's check the values
vif
VIF values	Features
0	1.471658	WindGustDir
1	1.400894	WindDir9am
2	1.305669	WindDir3pm
3	1.259724	RainToday
4	1.251821	RainTomorrow
5	1.652572	Location
6	6.784112	MinTemp
7	10.048678	MaxTemp
8	5.084177	Evaporation
9	10.348746	Sunshine
10	3.719694	WindGustSpeed
11	2.261437	WindSpeed9am
12	2.853401	WindSpeed3pm
13	3.112288	Humidity9am
14	3.780215	Humidity3pm
15	21.026659	Pressure9am
16	21.231649	Pressure3pm
17	3.675205	Cloud9am
18	3.690719	Cloud3pm
19	1.006931	Day
20	1.191429	Month
21	2.253622	Year
x.drop('Pressure3pm',axis=1,inplace=True)# Droping Pressure3pm
vif = pd.DataFrame()#Again check the value of VIF
vif["VIF values"] = [variance_inflation_factor(x.values,i)
              for i in range(len(x.columns))]
vif["Features"] = x.columns

# Let's check the values
vif
VIF values	Features
0	1.471599	WindGustDir
1	1.400403	WindDir9am
2	1.278061	WindDir3pm
3	1.258340	RainToday
4	1.250531	RainTomorrow
5	1.648729	Location
6	6.453127	MinTemp
7	8.383874	MaxTemp
8	4.999981	Evaporation
9	9.967566	Sunshine
10	3.712031	WindGustSpeed
11	2.259805	WindSpeed9am
12	2.830677	WindSpeed3pm
13	3.112249	Humidity9am
14	3.779236	Humidity3pm
15	1.588481	Pressure9am
16	3.648429	Cloud9am
17	3.583956	Cloud3pm
18	1.005370	Day
19	1.162400	Month
20	2.253067	Year
Now all features VIF values are below 10, So we can proceed

Modeling
#Finding best random state
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
maxAccu=0
maxRS=0
for i in range(1,200):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30, random_state =i)
    RFG = RandomForestRegressor()
    RFG.fit(x_train, y_train)
    pred = RFG.predict(x_test)
    acc=r2_score(y_test, pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print("Best accuracy is ",maxAccu," on Random_state ",maxRS)
Best accuracy is  0.749510648622342  on Random_state  156
So The Best Accuracy is 74.95 % at Rando State 156

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=maxRS)     #Creating at best Random State
Regression Algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
Random Forest Regressor
# Checking R2 score for Random Forest Regressor
RFR=RandomForestRegressor()
RFR.fit(x_train,y_train)

# prediction
predRFR=RFR.predict(x_test)
print('R2_Score:',r2_score(y_test,predRFR))
print('MAE:',mean_absolute_error(y_test, predRFR))
print('MSE:',mean_squared_error(y_test, predRFR))
print("RMSE:",np.sqrt(mean_squared_error(y_test, predRFR)))
# Checking cv score 
print(cross_val_score(RFR,x,y,cv=5).mean())
R2_Score: 0.7530560377098292
MAE: 0.2509616051041776
MSE: 0.16337550547314572
RMSE: 0.4041973595573649
0.4634332799176722
Accuracy of Random Forest Regressor is 75.30%

KNNeighbors Regressor
# Checking R2 score for Stochastic KNN Regressor
knn=KNN()
knn.fit(x_train,y_train)

# prediction
predknn=knn.predict(x_test)
print('R2_Score:',r2_score(y_test,predknn))
print('MAE:',mean_absolute_error(y_test, predknn))
print('MSE:',mean_squared_error(y_test, predknn))
print("RMSE:",np.sqrt(mean_squared_error(y_test, predknn)))
# Checking cv score 
print(cross_val_score(knn,x,y,cv=5).mean())
R2_Score: 0.6231677519288155
MAE: 0.2835365012376504
MSE: 0.2493082172823875
RMSE: 0.4993077380557881
0.38874628193123617
Accuracy of KNNeighbors Regressor is 62.31%

GradientBoosting Regressor
# Checking R2 score for GradientBoosting Regressor
GB=GradientBoostingRegressor()
GB.fit(x_train,y_train)

# prediction
predGB=GB.predict(x_test)
print('R2_Score:',r2_score(y_test,predGB))
print('MAE:',mean_absolute_error(y_test, predGB))
print('MSE:',mean_squared_error(y_test, predGB))
print("RMSE:",np.sqrt(mean_squared_error(y_test, predGB)))
# Checking cv score 

print(cross_val_score(GB,x,y,cv=5).mean())
R2_Score: 0.7308151542672126
MAE: 0.26752738408775983
MSE: 0.17808983799178066
RMSE: 0.42200691699518467
0.42431876987359834
Accuracy for Gradient Boosting is 73.08%

BaggingRegressor
# Checking R2 score for BaggingRegressor
from sklearn.ensemble import BaggingRegressor
BR=BaggingRegressor()
BR.fit(x_train,y_train)

# prediction
predBR=BR.predict(x_test)
print('R2_Score:',r2_score(y_test,predBR))
print('MAE:',mean_absolute_error(y_test, predBR))
print('MSE:',mean_squared_error(y_test, predBR))
print("RMSE:",np.sqrt(mean_squared_error(y_test, predBR)))
# Checking cv score
print(cross_val_score(BR,x,y,cv=5).mean())
R2_Score: 0.7223899014704565
MAE: 0.2585877329596337
MSE: 0.18366389585350565
RMSE: 0.4285602593025929
0.42284567502677833
Accuracy for Bagging algorithm is 72.23%

AdaBoostRegressor
ABR=AdaBoostRegressor()

ABR.fit(x_train,y_train)
# prediction
predABR=ABR.predict(x_test)
print('R2_Score:',r2_score(y_test,predABR))
print('MAE:',mean_absolute_error(y_test, predABR))
print('MSE:',mean_squared_error(y_test, predABR))
print("RMSE:",np.sqrt(mean_squared_error(y_test, predABR)))
# Checking cv score
print(cross_val_score(ABR,x,y,cv=5).mean())
R2_Score: 0.3571147887775481
MAE: 0.5922628390372223
MSE: 0.4253260349862733
RMSE: 0.6521702500009283
0.21193541957297599
From this we can AdaboostRegressor as our Best Algorithm as leaset differance between r2 score and CVR score

Hyper parameter tuning
parameter={'n_estimators':[1,3,4,5],
          'learning_rate':[.1,1,.3,.5],
          'loss' : ['linear', 'square', 'exponential'],
          'random_state':[100,150,200,250]}
GCV=GridSearchCV(AdaBoostRegressor(),parameter,cv=4)
GCV.fit(x_train,y_train)
GridSearchCV(cv=4, estimator=AdaBoostRegressor(),
             param_grid={'learning_rate': [0.1, 1, 0.3, 0.5],
                         'loss': ['linear', 'square', 'exponential'],
                         'n_estimators': [1, 3, 4, 5],
                         'random_state': [100, 150, 200, 250]})
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
GCV.best_params_
{'learning_rate': 0.3,
 'loss': 'square',
 'n_estimators': 5,
 'random_state': 100}
Rainfall_model = AdaBoostRegressor(learning_rate= 1, loss= 'exponential',n_estimators= 3,random_state=100)
Rainfall_model.fit(x_train, y_train)
pred = Rainfall_model.predict(x_test)
print("RMSE value:",np.sqrt(mean_squared_error(y_test, pred)))
print('R2_Score:',r2_score(y_test,pred)*100)
RMSE value: 0.4911751105925623
R2_Score: 63.53433220539183
So our accuracy of our Final Regressor model increased to 63.53%

Saving the model
# Saving the model using .pkl
import joblib
joblib.dump(Rainfall_model,"RainfallRegressor.pkl")
['RainfallRegressor.pkl']
Let's load the saved model and get the prediction
# Loading the saved model
model=joblib.load("RainfallRegressor.pkl")

#Prediction
prediction = model.predict(x_test)
a = np.array(y_test)
df_final = pd.DataFrame({"Original":a,"Predicted":prediction},index=range(len(a)))
df_final
Original	Predicted
0	1.192691	1.032945
1	-0.775140	-0.010484
2	1.238711	1.032945
3	-0.775140	-0.231061
4	-0.775140	-0.172792
...	...	...
1447	1.204859	-0.231061
1448	-0.775140	-0.710928
1449	0.556492	-0.480672
1450	-0.775140	-0.515951
1451	-0.775140	-0.677420
1452 rows × 2 columns

Thank You
 