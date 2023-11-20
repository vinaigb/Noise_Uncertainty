import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Read the text file into a data frame
df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store Recall values in a list
flicker_Recall = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            flicker_Recall.append(float(Recall))

#  Create a data frame from the list of Recall values
flicker_Recall_df = pd.DataFrame({'Recall': flicker_Recall})

print(flicker_Recall_df)



Gaussian_Recall = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            Gaussian_Recall.append(float(Recall))

#  Create a data frame from the list of Recall values
Gaussian_Recall_df = pd.DataFrame({'Recall': Gaussian_Recall})

print(Gaussian_Recall_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            saltnpepper_loss.append(float(Recall))

#  Create a data frame from the list of Recall values
saltnpepper_loss_df = pd.DataFrame({'Recall': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            multiplicative_loss.append(float(Recall))

#  Create a data frame from the list of Recall values
multiplicative_loss_df = pd.DataFrame({'Recall': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            colored_loss.append(float(Recall))

#  Create a data frame from the list of Recall values
colored_loss_df = pd.DataFrame({'Recall': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            periodic_loss.append(float(Recall))

#  Create a data frame from the list of Recall values
periodic_loss_df = pd.DataFrame({'Recall': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            onebyf_loss.append(float(Recall))

#  Create a data frame from the list of Recall values
onebyf_loss_df = pd.DataFrame({'Recall': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            brown_loss.append(float(Recall))

#  Create a data frame from the list of Recall values
brown_loss_df = pd.DataFrame({'Recall': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            uniform_loss.append(float(Recall))

#  Create a data frame from the list of Recall values
uniform_loss_df = pd.DataFrame({'Recall': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "Recall:" in line:
            _, Recall = line.strip().split(': ')
            impulse_loss.append(float(Recall))

#  Create a data frame from the list of Recall values
impulse_loss_df = pd.DataFrame({'Recall': impulse_loss})

print(impulse_loss_df)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

# List of noise types

noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Recall': noise}) for df, noise in zip(Recall_dfs, noise_types)], axis=1)

# Plotting the KDE plot with specified line colors
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5, palette=colors)
plt.xlabel('Recall')
plt.ylabel('Density')
plt.title('KDE: Recall of Different Noise Types using LightGBM')

# Create a custom legend with specified colors and rectangular symbols
legend_labels = [mpatches.Patch(color=colors[i], label=nt) for i, nt in enumerate(noise_types)]
plt.legend(handles=legend_labels, loc='upper left')

plt.grid(True)
plt.show()








# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Recall values for each noise type
mean_Recall_values = [df['Recall'].mean() for df in Recall_dfs]

# Creating a data frame for the mean Recall values
mean_Recall_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Recall': mean_Recall_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Recall_df['Noise Type'], mean_Recall_df['Mean Recall'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Recall')
plt.title('Mean Recall using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Recall_values) - 0.001, max(mean_Recall_values) + 0.001)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Recall values for each noise type
mean_Recall_values = [df['Recall'].mean() for df in Recall_dfs]

# Creating a data frame for the mean Recall values
mean_Recall_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Recall': mean_Recall_values})

# Sort the DataFrame by 'Mean Recall'
mean_Recall_df = mean_Recall_df.sort_values(by='Mean Recall')

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Recall_df['Noise Type'], mean_Recall_df['Mean Recall'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Recall')
plt.title('Mean Recall using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Recall_values) - 0.001, max(mean_Recall_values) + 0.001)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Recall values for each noise type
mean_Recall_values = [df['Recall'].mean() for df in Recall_dfs]

# Creating a data frame for the mean Recall values
mean_Recall_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Recall': mean_Recall_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_Recall_df['Noise Type'], mean_Recall_df['Mean Recall'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean Recall')
plt.title('Mean Recall using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, Recall_dfs):
    plt.plot(df['Recall'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('Recall')
plt.title('Recall using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Recall_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Recall', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('Recall')
plt.title('Recall using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



# List of noise types

noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']


# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Recall_dfs, noise_types)])

# Define a custom color palette for the noise_types
custom_palette = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Recall', data=combined_df, palette=custom_palette)
plt.xlabel('Noise Type')
plt.ylabel('Recall')
plt.title('Recall using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()







# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Recall': noise}) for df, noise in zip(Recall_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('Recall')
plt.ylabel('Density')
plt.title('KDE: Recall of Different Noise Types using LightGBM')
plt.legend(noise_types, loc='upper left')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(Recall_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['Recall'])
plt.xlabel('Noise Type')
plt.ylabel('Recall')
plt.title('Recall of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['Recall'] for df in Recall_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('Recall')
plt.title('Recall of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Recall data frames into a single data frame
combined_df = pd.concat(Recall_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Recall', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Recall')
plt.title('Recall of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Recall data frames into a single data frame
combined_df = pd.concat(Recall_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by 'Recall' values
sorted_df = combined_df.sort_values(by='Recall')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Recall', data=sorted_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Recall')
plt.title('Recall of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()




# Define a custom color palette for the noise_types
custom_palette = ['blue', 'purple', 'teal', 'brown', 'pink', 'orange', 'black', 'green', 'gray', 'red' ]


# Define a custom color palette for the noise_types
#custom_palette = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']

# List of noise types
noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

# List of data frames
Recall_dfs = [flicker_Recall_df, Gaussian_Recall_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Recall data frames into a single data frame
combined_df = pd.concat(Recall_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by 'Recall' values
sorted_df = combined_df.sort_values(by='Recall')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Recall', data=sorted_df, palette=custom_palette)
plt.xlabel('Noise Type')
plt.ylabel('Recall')
plt.title('Recall of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()









