import numpy as np

# Assuming you have three vectors with the same length
vector1 = np.array([1, 2, 3, 4])
vector2 = np.array([5, 6, 7, 8])
vector3 = np.array([9, 10, 11, 12])

# Calculate the element-wise average
average_vector = (vector1 + vector2 + vector3) / 3

# Print the result
print("Original Vectors:")
print("Vector 1:", vector1)
print("Vector 2:", vector2)
print("Vector 3:", vector3)
print("\nElement-wise Average:")
print("Average Vector:", average_vector)