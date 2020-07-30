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
<img src= "https://user-images.githubusercontent.com/66487971/88944290-38047e00-d295-11ea-8d38-c256a13d53bc.png" width = 800>

```python

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


```

<img src= "https://user-images.githubusercontent.com/66487971/88944442-68e4b300-d295-11ea-9191-77942cab6fb0.png" width = 700>


I notice there seems to be a private school with a graduation rate of higher than 100%. I am going to change it to 100%.

```python
df[df['Grad.Rate'] > 100]
```

<img src= "https://user-images.githubusercontent.com/66487971/88944835-e6a8be80-d295-11ea-8793-a386758905c6.png" width = 1000>

```python

df['Grad.Rate']['Cazenovia College'] = 100

```

Now I check the histogram again.

```python

sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

```

<img src= "https://user-images.githubusercontent.com/66487971/88945111-4010ed80-d296-11ea-8f71-58090c008314.png" width = 1000>

# K Means Cluster Creation

```python

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))
kmeans.cluster_centers_

```

<img src= "https://user-images.githubusercontent.com/66487971/88945268-7484a980-d296-11ea-97ff-a016cd3c101d.png" width = 500>

# Evaluation

There is no perfect way to evaluate clustering if you don't have the labels, however since this is just an exercise, I do have the labels, so I take advantage of this to evaluate omy clusters.

```python

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
       
df['Cluster'] = df['Private'].apply(converter)
df.head()
```

img src= "https://user-images.githubusercontent.com/66487971/88945518-c0375300-d296-11ea-9744-d17cc0f01072.png" width = 1000>

```python

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))

```

img src= "https://user-images.githubusercontent.com/66487971/88945637-e5c45c80-d296-11ea-9ceb-23f2148695c7.png" width = 450>

Not so bad considering the algorithm is purely using the features to cluster the universities into 2 distinct groups! 

# This concludes my project here. Thanks for reading all the way through.











