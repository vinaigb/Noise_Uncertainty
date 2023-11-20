import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store False Positive Rate values in a list
flicker_False_Positive_Rate = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            flicker_False_Positive_Rate.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
flicker_False_Positive_Rate_df = pd.DataFrame({'False Positive Rate': flicker_False_Positive_Rate})

print(flicker_False_Positive_Rate_df)



Gaussian_False_Positive_Rate = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            Gaussian_False_Positive_Rate.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
Gaussian_False_Positive_Rate_df = pd.DataFrame({'False Positive Rate': Gaussian_False_Positive_Rate})

print(Gaussian_False_Positive_Rate_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            saltnpepper_loss.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
saltnpepper_loss_df = pd.DataFrame({'False Positive Rate': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            multiplicative_loss.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
multiplicative_loss_df = pd.DataFrame({'False Positive Rate': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            colored_loss.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
colored_loss_df = pd.DataFrame({'False Positive Rate': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            periodic_loss.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
periodic_loss_df = pd.DataFrame({'False Positive Rate': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            onebyf_loss.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
onebyf_loss_df = pd.DataFrame({'False Positive Rate': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            brown_loss.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
brown_loss_df = pd.DataFrame({'False Positive Rate': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            uniform_loss.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
uniform_loss_df = pd.DataFrame({'False Positive Rate': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "False Positive Rate:" in line:
            _, False_Positive_Rate = line.strip().split(': ')
            impulse_loss.append(float(False_Positive_Rate))

#  Create a data frame from the list of False Positive Rate values
impulse_loss_df = pd.DataFrame({'False Positive Rate': impulse_loss})

print(impulse_loss_df)




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean False Positive Rate values for each noise type
mean_False_Positive_Rate_values = [df['False Positive Rate'].mean() for df in False_Positive_Rate_dfs]

# Creating a data frame for the mean False Positive Rate values
mean_False_Positive_Rate_df = pd.DataFrame({'Noise Type': noise_types, 'Mean False Positive Rate': mean_False_Positive_Rate_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_False_Positive_Rate_df['Noise Type'], mean_False_Positive_Rate_df['Mean False Positive Rate'])
plt.xlabel('Noise Type')
plt.ylabel('Mean False Positive Rate')
plt.title('Mean False Positive Rate using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean False Positive Rate values for each noise type
mean_False_Positive_Rate_values = [df['False Positive Rate'].mean() for df in False_Positive_Rate_dfs]

# Creating a data frame for the mean False Positive Rate values
mean_False_Positive_Rate_df = pd.DataFrame({'Noise Type': noise_types, 'Mean False Positive Rate': mean_False_Positive_Rate_values})

# Sort the DataFrame by 'Mean False Positive Rate'
mean_False_Positive_Rate_df = mean_False_Positive_Rate_df.sort_values(by='Mean False Positive Rate')

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_False_Positive_Rate_df['Noise Type'], mean_False_Positive_Rate_df['Mean False Positive Rate'])
plt.xlabel('Noise Type')
plt.ylabel('Mean False Positive Rate')
plt.title('Mean False Positive Rate using LightGBM for Different Noise Types')
plt.xticks(rotation=45)

plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean False Positive Rate values for each noise type
mean_False_Positive_Rate_values = [df['False Positive Rate'].mean() for df in False_Positive_Rate_dfs]

# Creating a data frame for the mean False Positive Rate values
mean_False_Positive_Rate_df = pd.DataFrame({'Noise Type': noise_types, 'Mean False Positive Rate': mean_False_Positive_Rate_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_False_Positive_Rate_df['Noise Type'], mean_False_Positive_Rate_df['Mean False Positive Rate'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean False Positive Rate')
plt.title('Mean False Positive Rate using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, False_Positive_Rate_dfs):
    plt.plot(df['False Positive Rate'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(False_Positive_Rate_dfs, noise_types)])


# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='False Positive Rate', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# List of noise types

noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']


# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(False_Positive_Rate_dfs, noise_types)])


# Define a custom color palette for the noise_types
custom_palette = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='False Positive Rate', data=combined_df, palette=custom_palette)
plt.xlabel('Noise Type')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'False Positive Rate': noise}) for df, noise in zip(False_Positive_Rate_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('False Positive Rate')
plt.ylabel('Density')
plt.title('KDE: False Positive Rate of Different Noise Types using LightGBM')
plt.legend(noise_types)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(False_Positive_Rate_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['False Positive Rate'])
plt.xlabel('Noise Type')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['False Positive Rate'] for df in False_Positive_Rate_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all False Positive Rate data frames into a single data frame
combined_df = pd.concat(False_Positive_Rate_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='False Positive Rate', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
False_Positive_Rate_dfs = [flicker_False_Positive_Rate_df, Gaussian_False_Positive_Rate_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all False Positive Rate data frames into a single data frame
combined_df = pd.concat(False_Positive_Rate_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Calculate median False Positive Rate for each noise type
median_False_Positive_Rate = combined_df.groupby('Noise Type')['False Positive Rate'].median().sort_values()

# Sort the DataFrame by median False Positive Rate
combined_df['Noise Type'] = pd.Categorical(combined_df['Noise Type'], categories=median_False_Positive_Rate.index, ordered=True)
combined_df = combined_df.sort_values(by='Noise Type')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='False Positive Rate', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()











