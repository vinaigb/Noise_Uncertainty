import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store Balanced Accuracy values in a list
flicker_balanced_accuracy = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            flicker_balanced_accuracy.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
flicker_balanced_accuracy_df = pd.DataFrame({'Balanced Accuracy': flicker_balanced_accuracy})

print(flicker_balanced_accuracy_df)



Gaussian_balanced_accuracy = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            Gaussian_balanced_accuracy.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
Gaussian_balanced_accuracy_df = pd.DataFrame({'Balanced Accuracy': Gaussian_balanced_accuracy})

print(Gaussian_balanced_accuracy_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            saltnpepper_loss.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
saltnpepper_loss_df = pd.DataFrame({'Balanced Accuracy': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            multiplicative_loss.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
multiplicative_loss_df = pd.DataFrame({'Balanced Accuracy': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            colored_loss.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
colored_loss_df = pd.DataFrame({'Balanced Accuracy': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            periodic_loss.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
periodic_loss_df = pd.DataFrame({'Balanced Accuracy': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            onebyf_loss.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
onebyf_loss_df = pd.DataFrame({'Balanced Accuracy': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            brown_loss.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
brown_loss_df = pd.DataFrame({'Balanced Accuracy': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            uniform_loss.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
uniform_loss_df = pd.DataFrame({'Balanced Accuracy': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "Balanced Accuracy:" in line:
            _, balanced_accuracy = line.strip().split(': ')
            impulse_loss.append(float(balanced_accuracy))

#  Create a data frame from the list of Balanced Accuracy values
impulse_loss_df = pd.DataFrame({'Balanced Accuracy': impulse_loss})

print(impulse_loss_df)



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

# List of noise types
noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Balanced Accuracy': noise}) for df, noise in zip(balanced_accuracy_dfs, noise_types)], axis=1)

# Plotting the KDE plot with specified line colors
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5, palette=colors)
plt.xlabel('Balanced Accuracy')
plt.ylabel('Density')
plt.title('KDE: Balanced Accuracy of Different Noise Types using LightGBM')

# Create a custom legend with specified colors and rectangular symbols
legend_labels = [mpatches.Patch(color=colors[i], label=nt) for i, nt in enumerate(noise_types)]
plt.legend(handles=legend_labels, loc='upper left')

plt.grid(True)
plt.show()









# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Balanced Accuracy values for each noise type
mean_balanced_accuracy_values = [df['Balanced Accuracy'].mean() for df in balanced_accuracy_dfs]

# Creating a data frame for the mean Balanced Accuracy values
mean_balanced_accuracy_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Balanced Accuracy': mean_balanced_accuracy_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_balanced_accuracy_df['Noise Type'], mean_balanced_accuracy_df['Mean Balanced Accuracy'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Balanced Accuracy')
plt.title('Mean Balanced Accuracy using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.show()


# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Balanced Accuracy values for each noise type
mean_balanced_accuracy_values = [df['Balanced Accuracy'].mean() for df in balanced_accuracy_dfs]

# Creating a data frame for the mean Balanced Accuracy values
mean_balanced_accuracy_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Balanced Accuracy': mean_balanced_accuracy_values})

# Plotting the bar plot with scaled Y-axis
plt.figure(figsize=(10, 6))
plt.bar(mean_balanced_accuracy_df['Noise Type'], mean_balanced_accuracy_df['Mean Balanced Accuracy'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Balanced Accuracy')
plt.title('Mean Balanced Accuracy using LightGBM for Different Noise Types')
plt.xticks(rotation=45)

# Scale the Y-axis to improve visibility
plt.ylim(min(mean_balanced_accuracy_values) - 0.001, max(mean_balanced_accuracy_values) + 0.001)

plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Balanced Accuracy values for each noise type
mean_balanced_accuracy_values = [df['Balanced Accuracy'].mean() for df in balanced_accuracy_dfs]

# Creating a data frame for the mean Balanced Accuracy values
mean_balanced_accuracy_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Balanced Accuracy': mean_balanced_accuracy_values})

# Sort the DataFrame by 'Mean Balanced Accuracy'
mean_balanced_accuracy_df = mean_balanced_accuracy_df.sort_values(by='Mean Balanced Accuracy')

# Plotting the bar plot with scaled Y-axis
plt.figure(figsize=(10, 6))
plt.bar(mean_balanced_accuracy_df['Noise Type'], mean_balanced_accuracy_df['Mean Balanced Accuracy'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Balanced Accuracy')
plt.title('Mean Balanced Accuracy using LightGBM for Different Noise Types')
plt.xticks(rotation=45)

# Scale the Y-axis to improve visibility
plt.ylim(min(mean_balanced_accuracy_values) - 0.001, max(mean_balanced_accuracy_values) + 0.001)

plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Balanced Accuracy values for each noise type
mean_balanced_accuracy_values = [df['Balanced Accuracy'].mean() for df in balanced_accuracy_dfs]

# Creating a data frame for the mean Balanced Accuracy values
mean_balanced_accuracy_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Balanced Accuracy': mean_balanced_accuracy_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_balanced_accuracy_df['Noise Type'], mean_balanced_accuracy_df['Mean Balanced Accuracy'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean Balanced Accuracy')
plt.title('Mean Balanced Accuracy using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, balanced_accuracy_dfs):
    plt.plot(df['Balanced Accuracy'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(balanced_accuracy_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Balanced Accuracy', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Balanced Accuracy': noise}) for df, noise in zip(balanced_accuracy_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('Balanced Accuracy')
plt.ylabel('Density')
plt.title('KDE: Balanced Accuracy of Different Noise Types using LightGBM')
plt.legend(noise_types, loc='upper left')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(balanced_accuracy_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['Balanced Accuracy'])
plt.xlabel('Noise Type')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['Balanced Accuracy'] for df in balanced_accuracy_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Balanced Accuracy data frames into a single data frame
combined_df = pd.concat(balanced_accuracy_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Balanced Accuracy', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
balanced_accuracy_dfs = [flicker_balanced_accuracy_df, Gaussian_balanced_accuracy_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Balanced Accuracy data frames into a single data frame
combined_df = pd.concat(balanced_accuracy_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Calculate median Balanced Accuracy for each noise type
median_balanced_accuracy = combined_df.groupby('Noise Type')['Balanced Accuracy'].median().sort_values()

# Sort the DataFrame by median Balanced Accuracy
combined_df['Noise Type'] = pd.Categorical(combined_df['Noise Type'], categories=median_balanced_accuracy.index, ordered=True)
combined_df = combined_df.sort_values(by='Noise Type')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Balanced Accuracy', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()











