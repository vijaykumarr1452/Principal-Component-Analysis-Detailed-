# Principal Component Analysis (PCA)

## What is PCA?
**Principal Component Analysis (PCA)** is a statistical technique used for dimensionality reduction. It transforms the original data into a new coordinate system where:
1. The first principal component accounts for the largest variance in the data.
2. The second principal component accounts for the next largest variance, and so on.

### Key Benefits
- **Dimensionality Reduction**: Reduces computational cost and eliminates redundant features.
- **Data Visualization**: Projects high-dimensional data into 2D or 3D for easy visualization.
- **Noise Reduction**: Helps remove irrelevant or noisy features.
- **Preprocessing for Machine Learning**: Reduces overfitting and speeds up training.

---

## Use Case: PCA On Breast Cancer Dataset

The Breast Cancer dataset is commonly used in machine learning for classification tasks (benign vs. malignant tumors).

### Dataset Description
- **Features**: Attributes of cell nuclei (e.g., mean radius, mean texture).
- **Target**: Tumor type (0 = malignant, 1 = benign).

---

## Steps to Apply PCA

1. **Load the Data**: Load the Breast Cancer dataset (e.g., from `sklearn.datasets`).
2. **Preprocess**: Normalize the data (PCA is sensitive to scale).
3. **Apply PCA**: Reduce dimensions while retaining most of the variance.
4. **Visualization**: Plot the first two principal components to visualize the data.
5. **Modeling**: Use reduced dimensions for training a classifier.

---

## Code Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot PCA components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Breast Cancer Dataset')
plt.colorbar(label='Target (0=Malignant, 1=Benign)')
plt.show()
```

---

## Plot of reduced 30 dimensions to 2 dimensions
![PNG](Screenshot from 2024-11-28 20-11-18.png)

---
### Observations

- **Visualization**: PCA reduces the dataset to 2 dimensions, making it easier to observe clusters or patterns (malignant vs. benign).
- **Modeling**: After PCA, the reduced data can be fed into models like logistic regression or SVM to classify tumors efficiently.

---

### Benefits

- **Speed**: Reduced dimensions make training faster.
- **Interpretability**: Highlights important variance in data.
- **Generalization**: Minimizes overfitting by removing irrelevant features.

---

## Connect :

If you have any questions or suggestions, feel free to reach out to me:

- Email: [vijaykumarit45@gmail.com](mailto:vijaykumarit45@gmail.com)
- GitHub: [Profile](https://github.com/vijaykumarr1452)
- Linkedin: [Linkedin](https://www.linkedin.com/in/rachuri-vijaykumar/)
- Twitter: [Twitter](https://x.com/vijay_viju1)


---
