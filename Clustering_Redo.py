#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import dbscan
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_excel('EastWestAirlines.xlsx', sheet_name='data') #Two sheets in the dataset
df.shape


# In[3]:


df.head()


# In[4]:


#Refactoring the dataset
df['Topflight']=np.where(df['Qual_miles']>0,1,0) #Condensing ambiguous dimensions into a more defined column
df['Total_miles'] = df['Balance'] + df['Bonus_miles'] + df['Flight_miles_12mo'] #A total overview of their customer miles
df['Total_trans'] = df['Flight_trans_12'] + df['Bonus_trans'] #Total overview of their transactional volume
df.drop(columns=['Qual_miles'], inplace=True)
df.head()


# In[5]:


df1=df.copy()
df1.head()


# In[6]:


def plot_one_vs_all(df, target_col, other_cols):
    
    n_plots = len(other_cols)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, col in enumerate(other_cols):
        if col != target_col:  # Skipping if same column
            axes[i].scatter(df1[target_col], df1[col], alpha=0.3)
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel(col)
            axes[i].set_title(f'{target_col} vs {col}')
            axes[i].grid(True, alpha=0.3)
            
            # Adding correlation tag to the scatter plot for easier identification of relationship
            corr = df1[target_col].corr(df1[col])
            axes[i].text(0.05, 0.95, f'r = {corr:.3f}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hiding unused subplots
    for j in range(len(other_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# In[7]:


plot_one_vs_all(df1, 'Flight_miles_12mo', ['Balance','Bonus_miles','Bonus_trans','Flight_trans_12','Topflight','Award?'])


# In[8]:


plot_one_vs_all(df1, 'Bonus_miles', ['Balance','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Topflight','Award?'])


# In[9]:


columns=['Balance','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Days_since_enroll','Total_miles','Total_trans']


# In[10]:


n=len(columns)
ncols = min(4, n)  # Limit to 4 columns for better readability
nrows = (n + ncols - 1) // ncols  # Calculate rows needed

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))

# Handle the case where axes might be 1D or 2D
if n == 1:
    axes = [axes]
elif nrows == 1:
    axes = axes.flatten()
else:
    axes = axes.flatten()

# Create box plots for each continuous column
for i, col in enumerate(columns):
    df1[col].plot(kind='box', vert=False,ax=axes[i])
    axes[i].set_title(f"{col} - Outlier Detection")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Values")  # Changed from "Frequency" to "Values" for box plots

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# In[11]:


outlier_columns=['Balance','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Total_miles','Total_trans']
for col in columns:
    q1=df1[col].quantile(0.25)
    q3=df1[col].quantile(0.75)
    iqr=q3-q1
    upper_whisker = q3 + 1.5 * iqr
    lower_whisker= q1-1.5*iqr
    df1[col] = np.where(df1[col] > upper_whisker, upper_whisker, df1[col])
    df1[col] = np.where(df1[col] < lower_whisker, lower_whisker, df1[col])


# In[12]:


n=len(columns)
ncols = min(4, n)  # Limit to 4 columns for better readability
nrows = (n + ncols - 1) // ncols  # Calculate rows needed

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))

# Handle the case where axes might be 1D or 2D
if n == 1:
    axes = [axes]
elif nrows == 1:
    axes = axes.flatten()
else:
    axes = axes.flatten()

# Create box plots for each continuous column
for i, col in enumerate(columns):
    df1[col].plot(kind='box', vert=False,ax=axes[i])
    axes[i].set_title(f"{col} - Outlier Handling")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Values")  
# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# In[13]:


SS=StandardScaler()
df1[columns]=SS.fit_transform(df1[columns])
df1.head()


# In[14]:


scores=[]
X=df1.iloc[:,1:]
for i in range(2,11,1):
    k_cluster=KMeans(n_clusters=i)
    df1['k_new']=k_cluster.fit_predict(X)
    scores.append(silhouette_score(X,df1['k_new']))
plt.scatter(range(2,11,1),scores)
plt.plot(range(2,11,1),scores, color='black')
plt.xlabel('KValue')
plt.ylabel('silhouette score')
plt.show()


# In[15]:


scores


# In[16]:


#New dataset for DBscan, no need to clear outliers for this clustering method
df2=df.copy()
df2


# In[17]:


cols=['Balance','Bonus_miles','Bonus_trans','Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll','Total_miles','Total_trans']
SS=StandardScaler()
df2[cols]=SS.fit_transform(df2[cols])
df2.head()


# In[18]:


X_db=df2.iloc[:,1:]
eps_values = np.arange(0.1, 1.0, 0.1)

min_samples_values = np.arange(5, 20, 1)

best_score = -1
best_params = {'eps': None, 'min_samples': None}


for eps in eps_values:
    for min_samples in min_samples_values:
        # Run DBSCAN
        DBSCAN = dbscan(X_db, eps=eps, min_samples=min_samples)
        labels = DBSCAN[1]

        # Skip if no clusters were found (only noise points)
        if len(set(labels)) > 1:
            # Calculate the silhouette score
            score = silhouette_score(X_db, labels)

            # Check if this is the best score so far
            if score > best_score:
                best_score = score
                best_params['eps'] = eps
                best_params['min_samples'] = min_samples

print("\nGrid Search Complete.")
print(f"Optimal eps: {best_params['eps']:.2f}")
print(f"Optimal min_samples: {best_params['min_samples']}")
print(f"Best Silhouette Score: {best_score:.4f}")


optimal_dbscan = dbscan(X, eps=best_params['eps'], min_samples=best_params['min_samples'])
df2['db_cluster']=optimal_dbscan[1]


# In[38]:


#Using the best performing clustern number from kmeans for agglomerative clustering
single_clusters=[]
complete_clusters=[]
avergae_clusters=[]
ward_clusters=[]
agg_single = AgglomerativeClustering(n_clusters=2, linkage='single')
single_clusters = agg_single.fit_predict(X)
agg_complete = AgglomerativeClustering(n_clusters=2, linkage='complete')
complete_clusters = agg_complete.fit_predict(X)
agg_average = AgglomerativeClustering(n_clusters=2, linkage='average')
average_clusters = agg_average.fit_predict(X)
agg_ward = AgglomerativeClustering(n_clusters=2, linkage='ward')
ward_clusters = agg_ward.fit_predict(X)
# Calculate silhouette scores using the correctly named cluster variables
SS1 = silhouette_score(X, single_clusters)
SS2 = silhouette_score(X, complete_clusters)
SS3 = silhouette_score(X, average_clusters)
SS4 = silhouette_score(X, ward_clusters)

# Print the scores
print(f"Single Linkage Silhouette Score: {SS1:.4f}")
print(f"Complete Linkage Silhouette Score: {SS2:.4f}")
print(f"Average Linkage Silhouette Score: {SS3:.4f}")#Best performing cluster
print(f"Ward Linkage Silhouette Score: {SS4:.4f}")


# In[19]:


X_agg=df1.iloc[:,1:14]
X_agg


# In[20]:


agg_scores=[]
for i in range(2,11,1):
    agg_average = AgglomerativeClustering(n_clusters=i, linkage='average')
    df1['agg_new']=agg_average.fit_predict(X_agg)
    agg_scores.append(silhouette_score(X_agg,df1['agg_new']))
ward=[]
for i in range(2,11,1):
    agg_ward = AgglomerativeClustering(n_clusters=i, linkage='ward')
    df1['ward_new']=agg_ward.fit_predict(X_agg)
    ward.append(silhouette_score(X_agg,df1['ward_new']))
complete=[]
for i in range(2,11,1):
    agg_complete = AgglomerativeClustering(n_clusters=i, linkage='complete')
    df1['complete_new']=agg_complete.fit_predict(X_agg)
    complete.append(silhouette_score(X_agg,df1['complete_new']))
single=[]
for i in range(2,11,1):
    agg_single = AgglomerativeClustering(n_clusters=i, linkage='single')
    df1['single_new']=agg_single.fit_predict(X_agg)
    single.append(silhouette_score(X_agg,df1['single_new']))

x = range(2, 11, 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of plots
axes = axes.flatten()  # make it easier to index

# Single linkage
axes[0].scatter(x, single)
axes[0].plot(x, single, color='black')
axes[0].set_title('Single Cluster Linkage')
axes[0].set_xlabel('ClusterValue')
axes[0].set_ylabel('Silhouette Score')

# Complete linkage
axes[1].scatter(x, complete)
axes[1].plot(x, complete, color='black')
axes[1].set_title('Complete Cluster Linkage')
axes[1].set_xlabel('ClusterValue')
axes[1].set_ylabel('Silhouette Score')

# Ward linkage
axes[2].scatter(x, ward)
axes[2].plot(x, ward, color='black')
axes[2].set_title('Ward Cluster Linkage')
axes[2].set_xlabel('ClusterValue')
axes[2].set_ylabel('Silhouette Score')

# Average linkage
axes[3].scatter(x, agg_scores)
axes[3].plot(x, agg_scores, color='black')
axes[3].set_title('Average Cluster Linkage')
axes[3].set_xlabel('ClusterValue')
axes[3].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()


# In[21]:


agg_cluster = AgglomerativeClustering(n_clusters=2, linkage='average')#2 clusters under average linkage method derived highes score
df['agg_cluster']=agg_cluster.fit_predict(X_agg)
df


# In[22]:


np.round(df['Bonus_miles'].groupby(df['agg_cluster']).mean())


# In[23]:


np.round(df['Bonus_trans'].groupby(df['agg_cluster']).sum())


# In[24]:


np.round(df['Total_trans'].groupby(df['agg_cluster']).sum())


# In[25]:


np.round(df['Flight_miles_12mo'].groupby(df['agg_cluster']).mean())


# In[26]:


np.round(df['Flight_trans_12'].groupby(df['agg_cluster']).sum())


# In[27]:


np.round(df['Total_miles'].groupby(df['agg_cluster']).mean())


# In[28]:


np.round(df['Balance'].groupby(df['agg_cluster']).mean())


# In[29]:


df['cc1_miles'].groupby(df['agg_cluster']).sum()


# In[30]:


df['cc2_miles'].groupby(df['agg_cluster']).sum()


# In[31]:


df['cc3_miles'].groupby(df['agg_cluster']).sum()


# In[32]:


df['Topflight'].groupby(df['agg_cluster']).value_counts()


# In[33]:


df['Award?'].groupby(df['agg_cluster']).value_counts()


# In[55]:


"""
Agglomerative/Hierarchical Clustering using average linkage was found to be a better fit for the data, as it achieved the 
highest silhouette score of all the methods. Under this method the most apt cluster number was 2, hence the customers can be 
divided into:

0 -> Infrequent/Inactive Flyers: Make up majority of the data but contribute the least in terms of miles covered. Their transaction
volume is high but volume per transaction is very low. 

1-> Frequent/Active Flyers: Make up a small part but contribute the highest in terms of miles covered. Though they have a lower
transaction volume the miles earned per transaction are much higher. They also have a higher mean balance. The individual in this
cluster might be high spenders.

KMeans was also a close second in terms of performance with a high 0.36 silhouette score at clusters=2, whereas DBScan was the 
worst performing with a silhouette score of 0.1 at 19 cluster value.
"""


# In[36]:


def plot_one_vs_all_clusters(df, target_col, other_cols):
    
    n_plots = len(other_cols)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, col in enumerate(other_cols):
        if col != target_col:
            axes[i].scatter(df[target_col], df[col], c=df['agg_cluster'], cmap='rainbow')
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel(col)
            axes[i].set_title(f'{target_col} vs {col}')
            axes[i].grid(True, alpha=0.3)
            
    # Hiding unused subplots
    for j in range(len(other_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# In[38]:


plot_one_vs_all_clusters(df, 'Bonus_miles', columns)


# In[ ]:




