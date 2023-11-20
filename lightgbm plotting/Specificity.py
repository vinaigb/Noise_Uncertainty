import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Read the text file into a data frame
df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store Specificity values in a list
flicker_Specificity = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            flicker_Specificity.append(float(Specificity))

#  Create a data frame from the list of Specificity values
flicker_Specificity_df = pd.DataFrame({'Specificity': flicker_Specificity})

print(flicker_Specificity_df)



Gaussian_Specificity = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            Gaussian_Specificity.append(float(Specificity))

#  Create a data frame from the list of Specificity values
Gaussian_Specificity_df = pd.DataFrame({'Specificity': Gaussian_Specificity})

print(Gaussian_Specificity_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            saltnpepper_loss.append(float(Specificity))

#  Create a data frame from the list of Specificity values
saltnpepper_loss_df = pd.DataFrame({'Specificity': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            multiplicative_loss.append(float(Specificity))

#  Create a data frame from the list of Specificity values
multiplicative_loss_df = pd.DataFrame({'Specificity': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            colored_loss.append(float(Specificity))

#  Create a data frame from the list of Specificity values
colored_loss_df = pd.DataFrame({'Specificity': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            periodic_loss.append(float(Specificity))

#  Create a data frame from the list of Specificity values
periodic_loss_df = pd.DataFrame({'Specificity': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            onebyf_loss.append(float(Specificity))

#  Create a data frame from the list of Specificity values
onebyf_loss_df = pd.DataFrame({'Specificity': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            brown_loss.append(float(Specificity))

#  Create a data frame from the list of Specificity values
brown_loss_df = pd.DataFrame({'Specificity': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            uniform_loss.append(float(Specificity))

#  Create a data frame from the list of Specificity values
uniform_loss_df = pd.DataFrame({'Specificity': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "Specificity:" in line:
            _, Specificity = line.strip().split(': ')
            impulse_loss.append(float(Specificity))

#  Create a data frame from the list of Specificity values
impulse_loss_df = pd.DataFrame({'Specificity': impulse_loss})

print(impulse_loss_df)



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

# List of noise types

noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Specificity': noise}) for df, noise in zip(Specificity_dfs, noise_types)], axis=1)

# Plotting the KDE plot with specified line colors
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5, palette=colors)
plt.xlabel('Specificity')
plt.ylabel('Density')
plt.title('KDE: Specificity of Different Noise Types using LightGBM')

# Create a custom legend with specified colors and rectangular symbols
legend_labels = [mpatches.Patch(color=colors[i], label=nt) for i, nt in enumerate(noise_types)]
plt.legend(handles=legend_labels, loc='upper left')

plt.grid(True)
plt.show()








# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Specificity values for each noise type
mean_Specificity_values = [df['Specificity'].mean() for df in Specificity_dfs]

# Creating a data frame for the mean Specificity values
mean_Specificity_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Specificity': mean_Specificity_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Specificity_df['Noise Type'], mean_Specificity_df['Mean Specificity'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Specificity')
plt.title('Mean Specificity using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.show()



## Y axis scaling

# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Specificity values for each noise type
mean_Specificity_values = [df['Specificity'].mean() for df in Specificity_dfs]

# Creating a data frame for the mean Specificity values
mean_Specificity_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Specificity': mean_Specificity_values})

# Plotting the bar plot with adjusted Y-axis scaling
plt.figure(figsize=(10, 6))
plt.bar(mean_Specificity_df['Noise Type'], mean_Specificity_df['Mean Specificity'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Specificity')
plt.title('Mean Specificity using LightGBM for Different Noise Types')
plt.xticks(rotation=45)

# Set a custom Y-axis range to make differences more visible
plt.ylim(min(mean_Specificity_values) - 0.001, max(mean_Specificity_values) + 0.001)

plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Specificity values for each noise type
mean_Specificity_values = [df['Specificity'].mean() for df in Specificity_dfs]

# Creating a data frame for the mean Specificity values
mean_Specificity_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Specificity': mean_Specificity_values})

# Sort the DataFrame by 'Mean Specificity'
mean_Specificity_df = mean_Specificity_df.sort_values(by='Mean Specificity')

# Plotting the bar plot with adjusted Y-axis scaling
plt.figure(figsize=(10, 6))
plt.bar(mean_Specificity_df['Noise Type'], mean_Specificity_df['Mean Specificity'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Specificity')
plt.title('Mean Specificity using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Specificity_values) - 0.001, max(mean_Specificity_values) + 0.001)

plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Specificity values for each noise type
mean_Specificity_values = [df['Specificity'].mean() for df in Specificity_dfs]

# Creating a data frame for the mean Specificity values
mean_Specificity_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Specificity': mean_Specificity_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_Specificity_df['Noise Type'], mean_Specificity_df['Mean Specificity'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean Specificity')
plt.title('Mean Specificity using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, Specificity_dfs):
    plt.plot(df['Specificity'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('Specificity')
plt.title('Specificity using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Specificity_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Specificity', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('Specificity')
plt.title('Specificity using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Specificity': noise}) for df, noise in zip(Specificity_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('Specificity')
plt.ylabel('Density')
plt.title('KDE: Specificity of Different Noise Types using LightGBM')
plt.legend(noise_types, loc='upper left')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(Specificity_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['Specificity'])
plt.xlabel('Noise Type')
plt.ylabel('Specificity')
plt.title('Specificity of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['Specificity'] for df in Specificity_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('Specificity')
plt.title('Specificity of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Specificity data frames into a single data frame
combined_df = pd.concat(Specificity_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Specificity', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Specificity')
plt.title('Specificity of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Specificity data frames into a single data frame
combined_df = pd.concat(Specificity_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Calculate median Specificity for each noise type
median_Specificity = combined_df.groupby('Noise Type')['Specificity'].median().sort_values()

# Sort the DataFrame by median Specificity
combined_df['Noise Type'] = pd.Categorical(combined_df['Noise Type'], categories=median_Specificity.index, ordered=True)
combined_df = combined_df.sort_values(by='Noise Type')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Specificity', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Specificity')
plt.title('Specificity of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# Define a custom color palette for the noise_types
custom_palette = ['blue', 'purple', 'teal', 'brown', 'pink', 'orange', 'green', 'black',  'gray', 'red' ]

# List of noise types
noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

# List of data frames
Specificity_dfs = [flicker_Specificity_df, Gaussian_Specificity_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Specificity data frames into a single data frame
combined_df = pd.concat(Specificity_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Calculate median Specificity for each noise type
median_Specificity = combined_df.groupby('Noise Type')['Specificity'].median().sort_values()

# Sort the DataFrame by median Specificity
combined_df['Noise Type'] = pd.Categorical(combined_df['Noise Type'], categories=median_Specificity.index, ordered=True)
combined_df = combined_df.sort_values(by='Noise Type')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Specificity', data=combined_df, palette=custom_palette)
plt.xlabel('Noise Type')
plt.ylabel('Specificity')
plt.title('Specificity of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()








