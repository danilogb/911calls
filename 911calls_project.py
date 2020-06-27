'''
> Solutions for 911 Calls EDA project
> by Danilo Brandao
> for Jose Portilla's Udemy Python for DS and ML course
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('911.csv')
df.info()
df.head()

# What are the top 5 zipcodes for 911 calls?
print('\nThe 5 zip codes with most 911 calls are:\n{}'.format(df['zip'].value_counts().head(5)))

# What are the top 5 townships (twp) for 911 calls?
print('\nThe 5 townships with most 911 calls are:\n{}'.format(df['twp'].value_counts().head(5)))

# How many unique title codes are there?
print('\nThere are {} unique emergency codes.'.format(len(df['title'].unique())))

# In the titles column there are "Reasons/Departments" specified before the title code. Create a new column called "Reason" that contains this string value.
df['Reason'] = df['title'].apply(lambda x:x.split(':')[0])
print('\nThe most common reasons for a 911 call are:\n{}'.format(df['Reason'].value_counts()))

# Countplot of 911 calls by Reason.
g = sns.countplot(df['Reason'])
g.set_title('Number of 911 calls by reason')

# What is the data type of the objects in the timeStamp column?
df['timeStamp'].apply(lambda x:type(x)).value_counts()

# Convert the column from strings to DateTime objects.
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# Create 3 new columns called 'Hour', 'Month', and 'Day of Week' based off of the timeStamp column.
df['Hour'] = df['timeStamp'].apply(lambda x:x.hour)
df['Month'] = df['timeStamp'].apply(lambda x:x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x:x.day_name())

# Create a countplot of the 'Day of Week' column with the hue based off of the 'Reason' column.
dayslist = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekchart = sns.countplot(x=df['Day of Week'], hue=df['Reason'], order=dayslist, palette='cubehelix')
weekchart.set_xticklabels(weekchart.get_xticklabels(), rotation=20)
weekchart.set_title('Frequency of 911 calls per weekday')

# Create a countplot of the 'Month' column with the hue based off of the 'Reason' column.
monthchart = sns.countplot(x=df['Month'], hue=df['Reason'], palette='cubehelix')
monthchart.set_title('Frequency of 911 calls per month')

# Uh-oh! Missing months! Time to fix it.
bymonth = df.groupby('Month').count()
bymonth.head()
bymonth['twp'].plot()

# Linear fit (Extremely simple solution, very likely not to be the most adequate. Done for simplicity and educational purposes.)
sns.lmplot(x='Month', y='twp', data=bymonth.reset_index())

# Create a new column called 'Date' that contains the date from the timeStamp column.
df['Date'] = df['timeStamp'].apply(lambda x:x.date())

# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.
dailychart = df.groupby('Date').count()
dailychart.head()
dailychart['twp'].plot()

# Create 3 separate plots with each plot representing a Reason for the 911 call.
# All setup in one figure so we can better compare the data.
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(df[df['Reason']=='Traffic'].groupby('Date').count()['twp'])
ax2.plot(df[df['Reason']=='EMS'].groupby('Date').count()['twp'])
ax3.plot(df[df['Reason']=='Fire'].groupby('Date').count()['twp'])
ax1.set_title('Traffic')
ax2.set_title('EMS')
ax3.set_title('Fire')
ax1.set_ylim(0,600)
ax2.set_ylim(0,600)
ax3.set_ylim(0,600)
ax1.tick_params(labelrotation=20)
ax2.tick_params(labelrotation=20)
ax3.tick_params(labelrotation=20)
fig.tight_layout()
plt.show()

# Onto maps!

# Restructure the dataframe so that the columns become the Hours and the Index becomes the days of the week.
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

# Create a heatmap
sns.heatmap(dayHour, cmap="viridis")

# Create a clustermap
sns.clustermap(dayHour, cmap="viridis")

# Same thing, now for months instead of hours.
dayMonth = df.groupby(['Day of Week', 'Month']).count()['Reason'].unstack()
sns.heatmap(dayMonth, cmap='viridis')  # Heatmap
sns.clustermap(dayMonth, cmap='viridis')  # Clustermap
plt.show()