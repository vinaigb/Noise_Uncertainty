import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Read the text file into a data frame
df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store log loss values in a list
flicker_log_loss = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            flicker_log_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
flicker_log_loss_df = pd.DataFrame({'Log Loss': flicker_log_loss})

print(flicker_log_loss_df)



Gaussian_log_loss = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            Gaussian_log_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
Gaussian_log_loss_df = pd.DataFrame({'Log Loss': Gaussian_log_loss})

print(Gaussian_log_loss_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            saltnpepper_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
saltnpepper_loss_df = pd.DataFrame({'Log Loss': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            multiplicative_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
multiplicative_loss_df = pd.DataFrame({'Log Loss': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            colored_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
colored_loss_df = pd.DataFrame({'Log Loss': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            periodic_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
periodic_loss_df = pd.DataFrame({'Log Loss': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            onebyf_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
onebyf_loss_df = pd.DataFrame({'Log Loss': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            brown_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
brown_loss_df = pd.DataFrame({'Log Loss': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            uniform_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
uniform_loss_df = pd.DataFrame({'Log Loss': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "Log Loss:" in line:
            _, log_loss = line.strip().split(': ')
            impulse_loss.append(float(log_loss))

#  Create a data frame from the list of log loss values
impulse_loss_df = pd.DataFrame({'Log Loss': impulse_loss})

print(impulse_loss_df)




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean log loss values for each noise type
mean_log_loss_values = [df['Log Loss'].mean() for df in log_loss_dfs]

# Creating a data frame for the mean log loss values
mean_log_loss_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Log Loss': mean_log_loss_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_log_loss_df['Noise Type'], mean_log_loss_df['Mean Log Loss'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Log Loss')
plt.title('Mean Log Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean log loss values for each noise type
mean_log_loss_values = [df['Log Loss'].mean() for df in log_loss_dfs]

# Creating a data frame for the mean log loss values
mean_log_loss_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Log Loss': mean_log_loss_values})

# Sort the data frame by mean Log Loss values
mean_log_loss_df = mean_log_loss_df.sort_values(by='Mean Log Loss')

# Plotting the bar plot with reordered X-axis labels
plt.figure(figsize=(10, 6))
plt.bar(mean_log_loss_df['Noise Type'], mean_log_loss_df['Mean Log Loss'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Log Loss')
plt.title('Mean Log Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean log loss values for each noise type
mean_log_loss_values = [df['Log Loss'].mean() for df in log_loss_dfs]

# Creating a data frame for the mean log loss values
mean_log_loss_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Log Loss': mean_log_loss_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_log_loss_df['Noise Type'], mean_log_loss_df['Mean Log Loss'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean Log Loss')
plt.title('Mean Log Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, log_loss_dfs):
    plt.plot(df['Log Loss'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('Log Loss')
plt.title('Log Loss using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(log_loss_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Log Loss', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('Log Loss')
plt.title('Log Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()





# List of noise types

noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']


# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(log_loss_dfs, noise_types)])

# Define a custom color palette for the noise_types
custom_palette = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'teal', 'black']

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Log Loss', data=combined_df, palette=custom_palette)
plt.xlabel('Noise Type')
plt.ylabel('Log Loss')
plt.title('Log Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Log Loss': noise}) for df, noise in zip(log_loss_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('Log Loss')
plt.ylabel('Density')
plt.title('KDE: Log Loss of Different Noise Types using LightGBM')
plt.legend(noise_types)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(log_loss_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['Log Loss'])
plt.xlabel('Noise Type')
plt.ylabel('Log Loss')
plt.title('Log Loss of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['Log Loss'] for df in log_loss_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('Log Loss')
plt.title('Log Loss of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all log loss data frames into a single data frame
combined_df = pd.concat(log_loss_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Log Loss', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Log Loss')
plt.title('Log Loss of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
log_loss_dfs = [flicker_log_loss_df, Gaussian_log_loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all log loss data frames into a single data frame
combined_df = pd.concat(log_loss_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by the median Log Loss values
sorted_df = combined_df.groupby('Noise Type')['Log Loss'].median().sort_values().index
combined_df['Noise Type'] = pd.Categorical(combined_df['Noise Type'], categories=sorted_df, ordered=True)

# Plotting the box plot using Seaborn with reordered X-axis labels
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Log Loss', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Log Loss')
plt.title('Log Loss of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()










