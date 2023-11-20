import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store Hamming Loss values in a list
flicker_Hamming_Loss = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            flicker_Hamming_Loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
flicker_Hamming_Loss_df = pd.DataFrame({'Hamming Loss': flicker_Hamming_Loss})

print(flicker_Hamming_Loss_df)



Gaussian_Hamming_Loss = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            Gaussian_Hamming_Loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
Gaussian_Hamming_Loss_df = pd.DataFrame({'Hamming Loss': Gaussian_Hamming_Loss})

print(Gaussian_Hamming_Loss_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            saltnpepper_loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
saltnpepper_loss_df = pd.DataFrame({'Hamming Loss': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            multiplicative_loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
multiplicative_loss_df = pd.DataFrame({'Hamming Loss': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            colored_loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
colored_loss_df = pd.DataFrame({'Hamming Loss': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            periodic_loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
periodic_loss_df = pd.DataFrame({'Hamming Loss': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            onebyf_loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
onebyf_loss_df = pd.DataFrame({'Hamming Loss': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            brown_loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
brown_loss_df = pd.DataFrame({'Hamming Loss': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            uniform_loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
uniform_loss_df = pd.DataFrame({'Hamming Loss': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "Hamming Loss:" in line:
            _, Hamming_Loss = line.strip().split(': ')
            impulse_loss.append(float(Hamming_Loss))

#  Create a data frame from the list of Hamming Loss values
impulse_loss_df = pd.DataFrame({'Hamming Loss': impulse_loss})

print(impulse_loss_df)




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Hamming Loss values for each noise type
mean_Hamming_Loss_values = [df['Hamming Loss'].mean() for df in Hamming_Loss_dfs]

# Creating a data frame for the mean Hamming Loss values
mean_Hamming_Loss_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Hamming Loss': mean_Hamming_Loss_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Hamming_Loss_df['Noise Type'], mean_Hamming_Loss_df['Mean Hamming Loss'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Hamming Loss')
plt.title('Mean Hamming Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Hamming Loss values for each noise type
mean_Hamming_Loss_values = [df['Hamming Loss'].mean() for df in Hamming_Loss_dfs]

# Creating a data frame for the mean Hamming Loss values
mean_Hamming_Loss_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Hamming Loss': mean_Hamming_Loss_values})

# Sort the DataFrame by mean Hamming Loss values
mean_Hamming_Loss_df = mean_Hamming_Loss_df.sort_values(by='Mean Hamming Loss')

# Plotting the bar plot with reordered X-axis labels
plt.figure(figsize=(10, 6))
plt.bar(mean_Hamming_Loss_df['Noise Type'], mean_Hamming_Loss_df['Mean Hamming Loss'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Hamming Loss')
plt.title('Mean Hamming Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.show()





# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Hamming Loss values for each noise type
mean_Hamming_Loss_values = [df['Hamming Loss'].mean() for df in Hamming_Loss_dfs]

# Creating a data frame for the mean Hamming Loss values
mean_Hamming_Loss_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Hamming Loss': mean_Hamming_Loss_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_Hamming_Loss_df['Noise Type'], mean_Hamming_Loss_df['Mean Hamming Loss'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean Hamming Loss')
plt.title('Mean Hamming Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, Hamming_Loss_dfs):
    plt.plot(df['Hamming Loss'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('Hamming Loss')
plt.title('Hamming Loss using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Hamming_Loss_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Hamming Loss', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('Hamming Loss')
plt.title('Hamming Loss using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Hamming Loss': noise}) for df, noise in zip(Hamming_Loss_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('Hamming Loss')
plt.ylabel('Density')
plt.title('KDE: Hamming Loss of Different Noise Types using LightGBM')
plt.legend(noise_types)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(Hamming_Loss_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['Hamming Loss'])
plt.xlabel('Noise Type')
plt.ylabel('Hamming Loss')
plt.title('Hamming Loss of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['Hamming Loss'] for df in Hamming_Loss_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('Hamming Loss')
plt.title('Hamming Loss of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Hamming Loss data frames into a single data frame
combined_df = pd.concat(Hamming_Loss_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Hamming Loss', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Hamming Loss')
plt.title('Hamming Loss of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Hamming_Loss_dfs = [flicker_Hamming_Loss_df, Gaussian_Hamming_Loss_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Hamming Loss data frames into a single data frame
combined_df = pd.concat(Hamming_Loss_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by the median Hamming Loss values
sorted_df = combined_df.groupby('Noise Type')['Hamming Loss'].median().sort_values().index
combined_df['Noise Type'] = pd.Categorical(combined_df['Noise Type'], categories=sorted_df, ordered=True)

# Plotting the box plot using Seaborn with reordered X-axis labels
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Hamming Loss', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Hamming Loss')
plt.title('Hamming Loss of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()










