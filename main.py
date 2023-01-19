import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("text_training.csv")
print(df.shape)
s = df.sum(axis=0)

print(s[s.values== 0])

# pca = PCA(n_components=2)
# pca.fit(df)
s = df.pop("rating")

pca = PCA(n_components=2)
pca.fit(df)

c = pca.components_.T[1:]
pca_df = pd.DataFrame(
    data=pca.components_[:2].T,
    columns=["PC1", "PC2"]
)

pca_df["target"] = df["rating"]

sns.set()
# sns.lmplot(
#     x="PC1",
#     y="PC2",
#     data=pca_df,
#     hue="target",
#     fit_reg=False,
#     legend=True
# )
sns.lmplot(
    x="PC1",
    y="target",
    data=pca_df,
    fit_reg=False,
    legend=True
)
plt.title("temp")
plt.show()

print("hi")