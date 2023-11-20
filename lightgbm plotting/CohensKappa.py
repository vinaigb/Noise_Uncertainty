import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Read the text file into a data frame
df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store Cohens Kappa values in a list
flicker_Cohens_Kappa = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            flicker_Cohens_Kappa.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
flicker_Cohens_Kappa_df = pd.DataFrame({'Cohens Kappa': flicker_Cohens_Kappa})

print(flicker_Cohens_Kappa_df)



Gaussian_Cohens_Kappa = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            Gaussian_Cohens_Kappa.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
Gaussian_Cohens_Kappa_df = pd.DataFrame({'Cohens Kappa': Gaussian_Cohens_Kappa})

print(Gaussian_Cohens_Kappa_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            saltnpepper_loss.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
saltnpepper_loss_df = pd.DataFrame({'Cohens Kappa': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            multiplicative_loss.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
multiplicative_loss_df = pd.DataFrame({'Cohens Kappa': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            colored_loss.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
colored_loss_df = pd.DataFrame({'Cohens Kappa': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            periodic_loss.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
periodic_loss_df = pd.DataFrame({'Cohens Kappa': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            onebyf_loss.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
onebyf_loss_df = pd.DataFrame({'Cohens Kappa': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            brown_loss.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
brown_loss_df = pd.DataFrame({'Cohens Kappa': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            uniform_loss.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
uniform_loss_df = pd.DataFrame({'Cohens Kappa': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "Cohens Kappa:" in line:
            _, Cohens_Kappa = line.strip().split(': ')
            impulse_loss.append(float(Cohens_Kappa))

#  Create a data frame from the list of Cohens Kappa values
impulse_loss_df = pd.DataFrame({'Cohens Kappa': impulse_loss})

print(impulse_loss_df)




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Cohens Kappa values for each noise type
mean_Cohens_Kappa_values = [df['Cohens Kappa'].mean() for df in Cohens_Kappa_dfs]

# Creating a data frame for the mean Cohens Kappa values
mean_Cohens_Kappa_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Cohens Kappa': mean_Cohens_Kappa_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Cohens_Kappa_df['Noise Type'], mean_Cohens_Kappa_df['Mean Cohens Kappa'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Cohens Kappa')
plt.title('Mean Cohens Kappa using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Cohens_Kappa_values) - 0.001, max(mean_Cohens_Kappa_values) + 0.001)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Cohens Kappa values for each noise type
mean_Cohens_Kappa_values = [df['Cohens Kappa'].mean() for df in Cohens_Kappa_dfs]

# Creating a data frame for the mean Cohens Kappa values
mean_Cohens_Kappa_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Cohens Kappa': mean_Cohens_Kappa_values})

# Sort the DataFrame by 'Mean Cohens Kappa'
mean_Cohens_Kappa_df = mean_Cohens_Kappa_df.sort_values(by='Mean Cohens Kappa')

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Cohens_Kappa_df['Noise Type'], mean_Cohens_Kappa_df['Mean Cohens Kappa'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Cohens Kappa')
plt.title('Mean Cohens Kappa using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Cohens_Kappa_values) - 0.001, max(mean_Cohens_Kappa_values) + 0.001)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Cohens Kappa values for each noise type
mean_Cohens_Kappa_values = [df['Cohens Kappa'].mean() for df in Cohens_Kappa_dfs]

# Creating a data frame for the mean Cohens Kappa values
mean_Cohens_Kappa_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Cohens Kappa': mean_Cohens_Kappa_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_Cohens_Kappa_df['Noise Type'], mean_Cohens_Kappa_df['Mean Cohens Kappa'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean Cohens Kappa')
plt.title('Mean Cohens Kappa using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, Cohens_Kappa_dfs):
    plt.plot(df['Cohens Kappa'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('Cohens Kappa')
plt.title('Cohens Kappa using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Cohens_Kappa_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Cohens Kappa', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('Cohens Kappa')
plt.title('Cohens Kappa using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Cohens Kappa': noise}) for df, noise in zip(Cohens_Kappa_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('Cohens Kappa')
plt.ylabel('Density')
plt.title('KDE: Cohens Kappa of Different Noise Types using LightGBM')
plt.legend(noise_types, loc='upper left')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(Cohens_Kappa_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['Cohens Kappa'])
plt.xlabel('Noise Type')
plt.ylabel('Cohens Kappa')
plt.title('Cohens Kappa of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['Cohens Kappa'] for df in Cohens_Kappa_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('Cohens Kappa')
plt.title('Cohens Kappa of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Cohens Kappa data frames into a single data frame
combined_df = pd.concat(Cohens_Kappa_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Cohens Kappa', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Cohens Kappa')
plt.title('Cohens Kappa of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Cohens Kappa data frames into a single data frame
combined_df = pd.concat(Cohens_Kappa_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by 'Cohens Kappa' values
sorted_df = combined_df.sort_values(by='Cohens Kappa')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Cohens Kappa', data=sorted_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Cohens Kappa')
plt.title('Cohens Kappa of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# Define a custom color palette for the noise_types
custom_palette = ['blue', 'purple', 'teal', 'brown', 'pink', 'orange', 'green', 'black',  'gray', 'red' ]

# List of noise types
noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

# List of data frames
Cohens_Kappa_dfs = [flicker_Cohens_Kappa_df, Gaussian_Cohens_Kappa_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Cohens Kappa data frames into a single data frame
combined_df = pd.concat(Cohens_Kappa_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by 'Cohens Kappa' values
sorted_df = combined_df.sort_values(by='Cohens Kappa')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Cohens Kappa', data=sorted_df, palette=custom_palette)
plt.xlabel('Noise Type')
plt.ylabel('Cohens Kappa')
plt.title('Cohens Kappa of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()








