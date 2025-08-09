# """
# this plots the class distribution of the dataset as a bar graph to look for class imbalance
# """

# import matplotlib.pyplot as plt

# plt.style.use("dark_background")


# # function to plot class distribution
# def plot_dataset(df):
#     # Class distribution
#     class_counts = df["class"].value_counts().sort_index()

#     # Log total number of files
#     total_files = len(df)
#     print(f"Total number of files: {total_files}")

#     plt.figure(figsize=(10, 5))
#     plt.bar(
#         class_counts.index,
#         class_counts.values,
#         width=0.9,
#         color="steelblue",
#         edgecolor="gray",
#     )
#     plt.title("Class Distribution")
#     plt.ylabel("Number of Samples")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
