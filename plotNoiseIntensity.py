
#Gaussian Noise
import pandas as pd
import matplotlib.pyplot as plt

# Initialize empty lists to store data
intensity_factors = []
accuracies = []
specificities = []
False_Positive_Rates = []
# Read data from the text file
with open("RF_gaussian.txt", "r") as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("Intensity Factor"):
            intensity_factor = float(lines[i].split()[-1])
            intensity_factors.append(intensity_factor)
        elif lines[i].startswith("Accuracy"):
            accuracy = float(lines[i].split()[-1])
            accuracies.append(accuracy)
        elif lines[i].startswith("Specificity"):
            specificity = float(lines[i].split()[-1])
            specificities.append(specificity)
        elif lines[i].startswith("False Positive Rate:"):
            False_Positive_Rate = float(lines[i].split()[-1])
            False_Positive_Rates.append(False_Positive_Rate)
        i += 1

# Create a data frame
df = pd.DataFrame({"Intensity Factor": intensity_factors, "Accuracy": accuracies, "Specificity":specificities, "False Positive Rate":False_Positive_Rates})

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df["Intensity Factor"], df["Accuracy"], marker='o', linestyle='-')
plt.title("LightGBM: Accuracy Noise Intensity Factor")
plt.xlabel("Flicker Noise Intensity Factor")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df["Intensity Factor"], df["Specificity"], marker='o', linestyle='-')
plt.title("LightGBM: Specificity vs Noise Intensity Factor")
plt.xlabel("Gaussian Noise Intensity Factor")
plt.ylabel("Specificity")
plt.grid(True)
plt.show()

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df["Intensity Factor"], df["False Positive Rate"], marker='o', linestyle='-')
plt.title("LightGBM: False Positive Rate vs Noise Intensity Factor")
plt.xlabel("Flicker Noise Intensity Factor")
plt.ylabel("False Positive Rate")
plt.grid(True)
plt.show()


############################################
