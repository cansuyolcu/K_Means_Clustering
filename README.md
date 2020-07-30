# K_Means_Clustering
For this project I will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.

It is very important to note, I actually have the labels for this data set, but I will NOT use them for the KMeans clustering algorithm, since that is an unsupervised learning algorithm. I will use the labels to try to get an idea of how well the algorithm performed.

# The Data
I will use a data frame with 777 observations on the following 18 variables.

- Private A factor with levels No and Yes indicating private or public university
- Apps Number of applications received
- Accept Number of applications accepted
- Enroll Number of new students enrolled
- Top10perc Pct. new students from top 10% of H.S. class
- Top25perc Pct. new students from top 25% of H.S. class
- F.Undergrad Number of fulltime undergraduates
- P.Undergrad Number of parttime undergraduates
- Outstate Out-of-state tuition
- Room.Board Room and board costs
- Books Estimated book costs
- Personal Estimated personal spending
- PhD Pct. of faculty with Ph.D.’s
- Terminal Pct. of faculty with terminal degree
- S.F.Ratio Student/faculty ratio
- perc.alumni Pct. alumni who donate
- Expend Instructional expenditure per student
- Grad.Rate Graduation rate

# Importing libraries

```python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

# Gettting the data

```python

df = pd.read_csv('College_Data',index_col=0)
df.head()
```

<img src= "https://user-images.githubusercontent.com/66487971/88943464-2ff80e80-d294-11ea-855c-9c801beea215.png" width = 1000>

```python
df.info()
```

<img src= "https://user-images.githubusercontent.com/66487971/88943655-6b92d880-d294-11ea-8158-1d34e4dc0e8e.png" width = 700>

```
```python
df.describe()
```

<img src= "https://user-images.githubusercontent.com/66487971/88943790-9715c300-d294-11ea-82a8-2d503be524ee.png" width = 1000>

# EDA

```python

sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
           
```

<img src= "https://user-images.githubusercontent.com/66487971/88943946-c9bfbb80-d294-11ea-8011-925179ad5cf4.png" width = 450>

```python
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
           
```           


<img src= "https://user-images.githubusercontent.com/66487971/88944163-11dede00-d295-11ea-834d-01334843095d.png" width = 450>

```python

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

```
<img src= "https://user-images.githubusercontent.com/66487971/88944290-38047e00-d295-11ea-8d38-c256a13d53bc.png" width = 600>

```python

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


```

img src= "https://user-images.githubusercontent.com/66487971/88944442-68e4b300-d295-11ea-9191-77942cab6fb0.png" width = 700>






