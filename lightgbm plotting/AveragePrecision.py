import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Read the text file into a data frame
df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store Average Precision values in a list
flicker_Average_Precision = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            flicker_Average_Precision.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
flicker_Average_Precision_df = pd.DataFrame({'Average Precision': flicker_Average_Precision})

print(flicker_Average_Precision_df)



Gaussian_Average_Precision = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            Gaussian_Average_Precision.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
Gaussian_Average_Precision_df = pd.DataFrame({'Average Precision': Gaussian_Average_Precision})

print(Gaussian_Average_Precision_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            saltnpepper_loss.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
saltnpepper_loss_df = pd.DataFrame({'Average Precision': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            multiplicative_loss.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
multiplicative_loss_df = pd.DataFrame({'Average Precision': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            colored_loss.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
colored_loss_df = pd.DataFrame({'Average Precision': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            periodic_loss.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
periodic_loss_df = pd.DataFrame({'Average Precision': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            onebyf_loss.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
onebyf_loss_df = pd.DataFrame({'Average Precision': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            brown_loss.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
brown_loss_df = pd.DataFrame({'Average Precision': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            uniform_loss.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
uniform_loss_df = pd.DataFrame({'Average Precision': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "Average Precision:" in line:
            _, Average_Precision = line.strip().split(': ')
            impulse_loss.append(float(Average_Precision))

#  Create a data frame from the list of Average Precision values
impulse_loss_df = pd.DataFrame({'Average Precision': impulse_loss})

print(impulse_loss_df)




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Average Precision values for each noise type
mean_Average_Precision_values = [df['Average Precision'].mean() for df in Average_Precision_dfs]

# Creating a data frame for the mean Average Precision values
mean_Average_Precision_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Average Precision': mean_Average_Precision_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Average_Precision_df['Noise Type'], mean_Average_Precision_df['Mean Average Precision'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Average Precision')
plt.title('Mean Average Precision using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Average_Precision_values) - 0.001, max(mean_Average_Precision_values) + 0.001)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Average Precision values for each noise type
mean_Average_Precision_values = [df['Average Precision'].mean() for df in Average_Precision_dfs]

# Creating a data frame for the mean Average Precision values
mean_Average_Precision_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Average Precision': mean_Average_Precision_values})

# Sort the DataFrame by 'Mean Average Precision'
mean_Average_Precision_df = mean_Average_Precision_df.sort_values(by='Mean Average Precision')

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Average_Precision_df['Noise Type'], mean_Average_Precision_df['Mean Average Precision'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Average Precision')
plt.title('Mean Average Precision using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Average_Precision_values) - 0.001, max(mean_Average_Precision_values) + 0.001)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Average Precision values for each noise type
mean_Average_Precision_values = [df['Average Precision'].mean() for df in Average_Precision_dfs]

# Creating a data frame for the mean Average Precision values
mean_Average_Precision_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Average Precision': mean_Average_Precision_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_Average_Precision_df['Noise Type'], mean_Average_Precision_df['Mean Average Precision'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean Average Precision')
plt.title('Mean Average Precision using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, Average_Precision_dfs):
    plt.plot(df['Average Precision'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('Average Precision')
plt.title('Average Precision using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Average_Precision_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Average Precision', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('Average Precision')
plt.title('Average Precision using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# List of noise types
noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Define a custom color palette for the noise_types
custom_palette = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Average_Precision_dfs, noise_types)])

# Plotting the Swarm Plot using the custom color palette
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Average Precision', data=combined_df, palette=custom_palette)
plt.xlabel('Noise Type')
plt.ylabel('Average Precision')
plt.title('Average Precision using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()





import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

# List of noise types
noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Average Precision': noise}) for df, noise in zip(Average_Precision_dfs, noise_types)], axis=1)

# Plotting the KDE plot with specified line colors
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5, palette=colors)
plt.xlabel('Average Precision')
plt.ylabel('Density')
plt.title('KDE: Average Precision of Different Noise Types using LightGBM')

# Create a custom legend with specified colors and rectangular symbols
legend_labels = [mpatches.Patch(color=colors[i], label=nt) for i, nt in enumerate(noise_types)]
plt.legend(handles=legend_labels, loc='upper left')

plt.grid(True)
plt.show()








# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(Average_Precision_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['Average Precision'])
plt.xlabel('Noise Type')
plt.ylabel('Average Precision')
plt.title('Average Precision of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['Average Precision'] for df in Average_Precision_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('Average Precision')
plt.title('Average Precision of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Average Precision data frames into a single data frame
combined_df = pd.concat(Average_Precision_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Average Precision', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Average Precision')
plt.title('Average Precision of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Average Precision data frames into a single data frame
combined_df = pd.concat(Average_Precision_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by 'Average Precision' values
sorted_df = combined_df.sort_values(by='Average Precision')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Average Precision', data=sorted_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Average Precision')
plt.title('Average Precision of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()













# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Average Precision': noise}) for df, noise in zip(Average_Precision_dfs, noise_types)], axis=1)
# combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Average_Precision_dfs, noise_types)])
# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('Average Precision')
plt.ylabel('Density')
plt.title('KDE: Average Precision of Different Noise Types using LightGBM')
plt.legend(noise_types, loc='upper left')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Average_Precision_dfs = [flicker_Average_Precision_df, Gaussian_Average_Precision_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Average Precision': noise}) for df, noise in zip(Average_Precision_dfs, noise_types)], axis=1)

# Plotting the KDE plot with specified line colors
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'black']
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5, palette=colors)
plt.xlabel('Average Precision')
plt.ylabel('Density')
plt.title('KDE: Average Precision of Different Noise Types using LightGBM')

# Create a custom legend with specified colors
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=nt, markerfacecolor=colors[i]) for i, nt in enumerate(noise_types)]
plt.legend(handles=legend_labels, loc='upper left')

plt.grid(True)
plt.show()












