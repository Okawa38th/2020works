__author__ = "Dachuan He"

import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('./vgsales.csv')
df = df.replace(np.nan, 0)
df.sort_values(by='Global_Sales', ascending=False, inplace=True)
df = df.iloc[:500, :]

df_num = df[['Rank', 'Critic_Score', 'User_Score', 'Total_Shipped', 'Global_Sales', 'NA_Sales', 'PAL_Sales', 'JP_Sales',
             'Other_Sales', 'Year']]

corr = df_num.corr().replace(np.nan, 0)

# 1. correlation matrix heatmap
fig = px.imshow(corr, color_continuous_scale=['blue', 'white', 'red'])
fig.show()

# 2. scatter plot
attr5 = abs(corr.sum()).sort_values(ascending=False)[0:5].index.tolist()
fig = px.scatter_matrix(df[attr5])
fig.show()

# 3. Parallel coordinates plot
x = StandardScaler().fit_transform(df_num)
df_std = pd.DataFrame(x, columns=df_num.columns)
sorted_cols = ['Global_Sales', 'PAL_Sales', 'Other_Sales', 'Rank', 'NA_Sales', 'Critic_Score', 'User_Score', 'JP_Sales',
               'Year', 'Total_Shipped']
df_sorted = df_std[sorted_cols]
fig = px.parallel_coordinates(df_sorted.iloc[:100, ])  # keep 100 observations
fig.show()

# 4. PCA with scree plot
pcaPlot = PCA(n_components=10)
pca = pcaPlot.fit_transform(df_std)

var_df = pd.DataFrame()
var_df['Component'] = range(1, len(pcaPlot.explained_variance_) + 1)
var_df['Explained Variance'] = pcaPlot.explained_variance_
var_df['Explained Variance Ratio'] = pcaPlot.explained_variance_ratio_

fig = px.line(x=var_df['Component'],
              y=np.cumsum(pcaPlot.explained_variance_),
              labels=dict(x="Component", y="Explained Variance"),
              title='Cumulative Explained Variance')
fig.add_bar(x=var_df['Component'], y=var_df['Explained Variance'])
fig.show()

# Scree plot
fig = px.line(var_df, x="Component", y="Explained Variance Ratio", title='Scree Plot')
fig.show()

# Scatter plot of PCA1 and PCA2
fig = px.scatter(x=pca[:, 0], y=pca[:, 1])
fig.update_layout(
    title_text='PCA plot (top 2 eigenvectors)'
)
fig.show()


# 5. PCA biplot.
def myplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='green', ha='center',
                     va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


myplot(pca[:, 0:2], np.transpose(pcaPlot.components_[0:2, :]), list(df_std.columns))

# 6. MDS of data
df_mds = df_num.iloc[:100, ]  # keep 100 observations
labels = df.iloc[:100, ]['ESRB_Rating'].tolist()

dis_matrix = euclidean_distances(df_mds)  # use top 100 observations

mds_model = manifold.MDS(n_components=2, random_state=123, dissimilarity='precomputed')
mds_fit = mds_model.fit(dis_matrix)
mds_coords = mds_model.fit_transform(dis_matrix)

mds_df = pd.DataFrame(mds_coords)
mds_df['ESRB_Rating'] = labels
mds_df.columns = ['dimension1', 'dimension2', 'ESRB_Rating']

fig = px.scatter(mds_df, x="dimension1", y="dimension2", color="ESRB_Rating")
fig.update_layout(
    title_text='MDS display of the data (use Euclidian distance)'
)
fig.show()


# 7. MDS of attributes
def one_minus_corr(corr):
    return 1 - abs(corr)


corr_matrix = corr.applymap(one_minus_corr)

mds_model = manifold.MDS(n_components=2,
                         random_state=123,
                         dissimilarity='precomputed')
mds_fit = mds_model.fit(corr_matrix)
mds_coords = mds_model.fit_transform(corr_matrix)  # shape is (10,2)

mds_df = pd.DataFrame(mds_coords)
mds_df['Attribute'] = corr_matrix.columns
mds_df.columns = ['First Dimension', 'Second Dimension', 'Attribute']

fig = px.scatter(mds_df, x="First Dimension", y="Second Dimension", text="Attribute")
fig.update_traces(textposition='top center')
fig.update_layout(
    title_text='MDS display of the attributes (use 1-|correlation| distance)'
)
fig.show()

plt.show()
