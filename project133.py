import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import pandas as pd
from sklearn.cluster import KMeans


data = pd.read_csv("pro-final.csv")

x = data.iloc[:,[3,4]].values

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append((kmeans.inertia_))

plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel('Number of clusters')
plt.show()

