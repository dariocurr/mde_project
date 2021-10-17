# mde_project

Empirical study to determine the relationships present between four different visualization techniques:

- 2D scatterplot for two-dimensional data
- 2D scatterplot for three-dimensional data
- interactive 3D scatterplot
- Draftman plot 

and four different dimensionality reduction techniques: 

- PCA
- PCA kernel
- MDS
- t-SNE

The analysis was carried out on a virtual dataset containing biomedical data used for the diagnosis of celiac disease and two classes: celiac and non-celiac.
From the graphical representations we can see that PCA and kernel PCA are more efficient for the visualization of the two classes. 
Moreover, it is noted that in most cases the 2D scatterplots are already sufficient to detect a good distinction between the classes. 
Decision trees have been created both for the unreduced dataset and for the datasets reduced by the previously mentioned techniques.
Finally, the results obtained were compared in terms of accuracy
