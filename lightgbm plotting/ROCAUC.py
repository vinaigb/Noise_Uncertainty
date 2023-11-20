import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Read the text file into a data frame
df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store ROC AUC values in a list
flicker_ROC_AUC = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            flicker_ROC_AUC.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
flicker_ROC_AUC_df = pd.DataFrame({'ROC AUC': flicker_ROC_AUC})

print(flicker_ROC_AUC_df)



Gaussian_ROC_AUC = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            Gaussian_ROC_AUC.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
Gaussian_ROC_AUC_df = pd.DataFrame({'ROC AUC': Gaussian_ROC_AUC})

print(Gaussian_ROC_AUC_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            saltnpepper_loss.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
saltnpepper_loss_df = pd.DataFrame({'ROC AUC': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            multiplicative_loss.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
multiplicative_loss_df = pd.DataFrame({'ROC AUC': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            colored_loss.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
colored_loss_df = pd.DataFrame({'ROC AUC': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            periodic_loss.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
periodic_loss_df = pd.DataFrame({'ROC AUC': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            onebyf_loss.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
onebyf_loss_df = pd.DataFrame({'ROC AUC': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            brown_loss.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
brown_loss_df = pd.DataFrame({'ROC AUC': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            uniform_loss.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
uniform_loss_df = pd.DataFrame({'ROC AUC': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "ROC AUC:" in line:
            _, ROC_AUC = line.strip().split(': ')
            impulse_loss.append(float(ROC_AUC))

#  Create a data frame from the list of ROC AUC values
impulse_loss_df = pd.DataFrame({'ROC AUC': impulse_loss})

print(impulse_loss_df)




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean ROC AUC values for each noise type
mean_ROC_AUC_values = [df['ROC AUC'].mean() for df in ROC_AUC_dfs]

# Creating a data frame for the mean ROC AUC values
mean_ROC_AUC_df = pd.DataFrame({'Noise Type': noise_types, 'Mean ROC AUC': mean_ROC_AUC_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_ROC_AUC_df['Noise Type'], mean_ROC_AUC_df['Mean ROC AUC'])
plt.xlabel('Noise Type')
plt.ylabel('Mean ROC AUC')
plt.title('Mean ROC AUC using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_ROC_AUC_values) - 0.001, max(mean_ROC_AUC_values) + 0.001)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean ROC AUC values for each noise type
mean_ROC_AUC_values = [df['ROC AUC'].mean() for df in ROC_AUC_dfs]

# Creating a data frame for the mean ROC AUC values
mean_ROC_AUC_df = pd.DataFrame({'Noise Type': noise_types, 'Mean ROC AUC': mean_ROC_AUC_values})

# Sort the DataFrame by 'Mean ROC AUC'
mean_ROC_AUC_df = mean_ROC_AUC_df.sort_values(by='Mean ROC AUC')

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_ROC_AUC_df['Noise Type'], mean_ROC_AUC_df['Mean ROC AUC'])
plt.xlabel('Noise Type')
plt.ylabel('Mean ROC AUC')
plt.title('Mean ROC AUC using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_ROC_AUC_values) - 0.001, max(mean_ROC_AUC_values) + 0.001)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean ROC AUC values for each noise type
mean_ROC_AUC_values = [df['ROC AUC'].mean() for df in ROC_AUC_dfs]

# Creating a data frame for the mean ROC AUC values
mean_ROC_AUC_df = pd.DataFrame({'Noise Type': noise_types, 'Mean ROC AUC': mean_ROC_AUC_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_ROC_AUC_df['Noise Type'], mean_ROC_AUC_df['Mean ROC AUC'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean ROC AUC')
plt.title('Mean ROC AUC using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, ROC_AUC_dfs):
    plt.plot(df['ROC AUC'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('ROC AUC')
plt.title('ROC AUC using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(ROC_AUC_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='ROC AUC', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('ROC AUC')
plt.title('ROC AUC using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'ROC AUC': noise}) for df, noise in zip(ROC_AUC_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('ROC AUC')
plt.ylabel('Density')
plt.title('KDE: ROC AUC of Different Noise Types using LightGBM')
plt.legend(noise_types, loc='upper left')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(ROC_AUC_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['ROC AUC'])
plt.xlabel('Noise Type')
plt.ylabel('ROC AUC')
plt.title('ROC AUC of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['ROC AUC'] for df in ROC_AUC_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('ROC AUC')
plt.title('ROC AUC of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all ROC AUC data frames into a single data frame
combined_df = pd.concat(ROC_AUC_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='ROC AUC', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('ROC AUC')
plt.title('ROC AUC of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
ROC_AUC_dfs = [flicker_ROC_AUC_df, Gaussian_ROC_AUC_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all ROC AUC data frames into a single data frame
combined_df = pd.concat(ROC_AUC_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by 'ROC AUC' values
sorted_df = combined_df.sort_values(by='ROC AUC')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='ROC AUC', data=sorted_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('ROC AUC')
plt.title('ROC AUC of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()













