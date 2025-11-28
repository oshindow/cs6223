import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix data
matrix = np.array([
    [5382, 386 ],
    [5895, 915 ]
])

# Percentages
percent = matrix / matrix.sum() * 100

# Combine number + percentage for display
labels = np.array([[
    f"{matrix[i, j]}\n({percent[i, j]:.2f}%)" for j in range(2)
] for i in range(2)])

plt.figure(figsize=(6, 5))
ax = sns.heatmap(
    matrix, 
    annot=labels, 
    fmt="", 
    cmap="YlGn", 
    cbar=True
)

# Title and axis labels
# plt.title("Confusion Matrix")
plt.xlabel("Property Test Case (Advanced)")
plt.ylabel("Result")

# Tick labels
ax.set_xticklabels(["Pass", "Fail"], fontsize=10)
ax.set_yticklabels(["Correct Result", "Wrong Result"], fontsize=10, rotation=90)

plt.tight_layout()
# plt.show()
plt.savefig("confusion_matrix.png")