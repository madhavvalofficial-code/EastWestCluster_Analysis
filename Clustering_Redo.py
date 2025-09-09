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


# In[31]:


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


# In[32]:


plot_one_vs_all(df1, 'Flight_miles_12mo', ['Balance','Bonus_miles','Bonus_trans','Flight_trans_12','Topflight','Award?'])


# In[33]:


plot_one_vs_all(df1, 'Bonus_miles', ['Balance','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Topflight','Award?'])


# In[6]:


columns=['Balance','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Days_since_enroll','Total_miles','Total_trans']


# In[7]:


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


# In[8]:


outlier_columns=['Balance','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Total_miles','Total_trans']
for col in columns:
    q1=df1[col].quantile(0.25)
    q3=df1[col].quantile(0.75)
    iqr=q3-q1
    upper_whisker = q3 + 1.5 * iqr
    lower_whisker= q1-1.5*iqr
    df1[col] = np.where(df1[col] > upper_whisker, upper_whisker, df1[col])
    df1[col] = np.where(df1[col] < lower_whisker, lower_whisker, df1[col])


# In[9]:


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


# In[10]:


SS=StandardScaler()
df1[columns]=SS.fit_transform(df1[columns])
df1.head()


# In[31]:


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


# In[41]:


#performing kclustering without Balance
scores=[]
X_kmeans=df1.iloc[:,1:14]
for i in range(2,11,1):
    k_cluster=KMeans(n_clusters=i)
    df1['k_new1']=k_cluster.fit_predict(X_kmeans)
    scores.append(silhouette_score(X_kmeans,df1['k_new1']))
plt.scatter(range(2,11,1),scores)
plt.plot(range(2,11,1),scores, color='black')
plt.xlabel('KValue')
plt.ylabel('silhouette score')
plt.show()



# In[42]:


scores


# In[16]:


df2=df.copy()
df2


# In[17]:


cols=['Balance','Bonus_miles','Bonus_trans','Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll','Total_miles','Total_trans']
SS=StandardScaler()
df2[cols]=SS.fit_transform(df2[cols])
df2.head()


# In[21]:


X_db=df2.iloc[:,1:]
eps_values = np.arange(0.1, 1.0, 0.1)

min_samples_values = np.arange(5, 20, 1)

best_score = -1
best_params = {'eps': None, 'min_samples': None}


for eps in eps_values:
    for min_samples in min_samples_values:
        # Run DBSCAN
        DBSCAN = dbscan(X, eps=eps, min_samples=min_samples)
        labels = DBSCAN[1]

        # Skip if no clusters were found (only noise points)
        if len(set(labels)) > 1:
            # Calculate the silhouette score
            score = silhouette_score(X, labels)

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
single_clusters = agg_single.fit_predict(X_kmeans)
agg_complete = AgglomerativeClustering(n_clusters=2, linkage='complete')
complete_clusters = agg_complete.fit_predict(X_kmeans)
agg_average = AgglomerativeClustering(n_clusters=2, linkage='average')
average_clusters = agg_average.fit_predict(X_kmeans)
agg_ward = AgglomerativeClustering(n_clusters=2, linkage='ward')
ward_clusters = agg_ward.fit_predict(X_kmeans)
# Calculate silhouette scores using the correctly named cluster variables
SS1 = silhouette_score(X_kmeans, single_clusters)
SS2 = silhouette_score(X_kmeans, complete_clusters)
SS3 = silhouette_score(X_kmeans, average_clusters)
SS4 = silhouette_score(X_kmeans, ward_clusters)

# Print the scores
print(f"Single Linkage Silhouette Score: {SS1:.4f}")
print(f"Complete Linkage Silhouette Score: {SS2:.4f}")
print(f"Average Linkage Silhouette Score: {SS3:.4f}")
print(f"Ward Linkage Silhouette Score: {SS4:.4f}")


# In[43]:


df1.head()


# In[44]:


agg_scores=[]
X_agg=df1.iloc[:,1:14]
for i in range(2,11,1):
    agg_average = AgglomerativeClustering(n_clusters=i, linkage='average')
    df1['agg_new']=agg_average.fit_predict(X_agg)
    agg_scores.append(silhouette_score(X_agg,df1['agg_new']))
plt.scatter(range(2,11,1),agg_scores)
plt.plot(range(2,11,1),agg_scores, color='black')
plt.xlabel('ClusterValue')
plt.ylabel('silhouette score')
plt.show()


# In[45]:


agg_cluster = AgglomerativeClustering(n_clusters=2, linkage='average')
df['agg_cluster']=agg_cluster.fit_predict(X_agg)
df


# In[58]:


df['ID#'].groupby(df['agg_cluster']).count()


# In[46]:


np.round(df['Bonus_miles'].groupby(df['agg_cluster']).mean())


# In[47]:


np.round(df['Bonus_trans'].groupby(df['agg_cluster']).sum())


# In[48]:


np.round(df['Total_trans'].groupby(df['agg_cluster']).sum())


# In[49]:


np.round(df['Flight_miles_12mo'].groupby(df['agg_cluster']).mean())


# In[50]:


np.round(df['Flight_trans_12'].groupby(df['agg_cluster']).sum())


# In[51]:


np.round(df['Total_miles'].groupby(df['agg_cluster']).mean())


# In[52]:


df['cc1_miles'].groupby(df['agg_cluster']).sum()


# In[53]:


df['cc2_miles'].groupby(df['agg_cluster']).sum()


# In[54]:


df['cc3_miles'].groupby(df['agg_cluster']).sum()


# In[55]:


df['Topflight'].groupby(df['agg_cluster']).value_counts()


# In[56]:


df['Award?'].groupby(df['agg_cluster']).value_counts()


# In[57]:


def plot_one_vs_all_clusters(df, target_col, other_cols):
    
    n_plots = len(other_cols)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, col in enumerate(other_cols):
        if col != target_col:
            axes[i].scatter(df[target_col], df[col], c=df['agg_cluster'], cmap='rainbow', alpha=0.3)
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel(col)
            axes[i].set_title(f'{target_col} vs {col}')
            axes[i].grid(True, alpha=0.3)
            
    # Hiding unused subplots
    for j in range(len(other_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()




# In[58]:


plot_one_vs_all_clusters(df, 'Bonus_miles', ['Balance','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Topflight','Award?'])
plot_one_vs_all_clusters(df1, 'Flight_miles_12mo', ['Balance','Bonus_trans','Bonus_miles','Flight_trans_12','Topflight','Award?'])




