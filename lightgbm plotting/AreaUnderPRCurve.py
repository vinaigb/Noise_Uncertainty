import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Read the text file into a data frame
df = pd.read_csv('flicker.txt')

print(df)


# Read the text file line by line and store Area Under PR Curve values in a list
flicker_Area_Under_PR_Curve = []
with open('flicker.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            flicker_Area_Under_PR_Curve.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
flicker_Area_Under_PR_Curve_df = pd.DataFrame({'Area Under PR Curve': flicker_Area_Under_PR_Curve})

print(flicker_Area_Under_PR_Curve_df)



Gaussian_Area_Under_PR_Curve = []
with open('Gaussian.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            Gaussian_Area_Under_PR_Curve.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
Gaussian_Area_Under_PR_Curve_df = pd.DataFrame({'Area Under PR Curve': Gaussian_Area_Under_PR_Curve})

print(Gaussian_Area_Under_PR_Curve_df)



saltnpepper_loss = []
with open('saltnpepper.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            saltnpepper_loss.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
saltnpepper_loss_df = pd.DataFrame({'Area Under PR Curve': saltnpepper_loss})

print(saltnpepper_loss_df)



multiplicative_loss = []
with open('multiplicative.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            multiplicative_loss.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
multiplicative_loss_df = pd.DataFrame({'Area Under PR Curve': multiplicative_loss})

print(multiplicative_loss_df)



colored_loss = []
with open('colored.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            colored_loss.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
colored_loss_df = pd.DataFrame({'Area Under PR Curve': colored_loss})

print(colored_loss_df)


periodic_loss = []
with open('periodic.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            periodic_loss.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
periodic_loss_df = pd.DataFrame({'Area Under PR Curve': periodic_loss})

print(periodic_loss_df)


onebyf_loss = []
with open('onebyf.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            onebyf_loss.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
onebyf_loss_df = pd.DataFrame({'Area Under PR Curve': onebyf_loss})

print(onebyf_loss_df)


brown_loss = []
with open('brown.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            brown_loss.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
brown_loss_df = pd.DataFrame({'Area Under PR Curve': brown_loss})

print(brown_loss_df)


uniform_loss = []
with open('uniform.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            uniform_loss.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
uniform_loss_df = pd.DataFrame({'Area Under PR Curve': uniform_loss})

print(uniform_loss_df)


impulse_loss = []
with open('impulse.txt', 'r') as file:
    for line in file:
        if "Area Under PR Curve:" in line:
            _, Area_Under_PR_Curve = line.strip().split(': ')
            impulse_loss.append(float(Area_Under_PR_Curve))

#  Create a data frame from the list of Area Under PR Curve values
impulse_loss_df = pd.DataFrame({'Area Under PR Curve': impulse_loss})

print(impulse_loss_df)




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Area Under PR Curve values for each noise type
mean_Area_Under_PR_Curve_values = [df['Area Under PR Curve'].mean() for df in Area_Under_PR_Curve_dfs]

# Creating a data frame for the mean Area Under PR Curve values
mean_Area_Under_PR_Curve_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Area Under PR Curve': mean_Area_Under_PR_Curve_values})

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Area_Under_PR_Curve_df['Noise Type'], mean_Area_Under_PR_Curve_df['Mean Area Under PR Curve'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Area Under PR Curve')
plt.title('Mean Area Under PR Curve using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Area_Under_PR_Curve_values) - 0.001, max(mean_Area_Under_PR_Curve_values) + 0.001)
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Area Under PR Curve values for each noise type
mean_Area_Under_PR_Curve_values = [df['Area Under PR Curve'].mean() for df in Area_Under_PR_Curve_dfs]

# Creating a data frame for the mean Area Under PR Curve values
mean_Area_Under_PR_Curve_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Area Under PR Curve': mean_Area_Under_PR_Curve_values})

# Sort the DataFrame by 'Mean Area Under PR Curve'
mean_Area_Under_PR_Curve_df = mean_Area_Under_PR_Curve_df.sort_values(by='Mean Area Under PR Curve')

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(mean_Area_Under_PR_Curve_df['Noise Type'], mean_Area_Under_PR_Curve_df['Mean Area Under PR Curve'])
plt.xlabel('Noise Type')
plt.ylabel('Mean Area Under PR Curve')
plt.title('Mean Area Under PR Curve using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.ylim(min(mean_Area_Under_PR_Curve_values) - 0.001, max(mean_Area_Under_PR_Curve_values) + 0.001)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Extracting the mean Area Under PR Curve values for each noise type
mean_Area_Under_PR_Curve_values = [df['Area Under PR Curve'].mean() for df in Area_Under_PR_Curve_dfs]

# Creating a data frame for the mean Area Under PR Curve values
mean_Area_Under_PR_Curve_df = pd.DataFrame({'Noise Type': noise_types, 'Mean Area Under PR Curve': mean_Area_Under_PR_Curve_values})

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(mean_Area_Under_PR_Curve_df['Noise Type'], mean_Area_Under_PR_Curve_df['Mean Area Under PR Curve'], marker='o', linestyle='--', color='b')
plt.xlabel('Noise Type')
plt.ylabel('Mean Area Under PR Curve')
plt.title('Mean Area Under PR Curve using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', 'onebyf', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the line graph
plt.figure(figsize=(20, 10))
for noise, df in zip(noise_types, Area_Under_PR_Curve_dfs):
    plt.plot(df['Area Under PR Curve'], label=noise, linewidth=3.5)

plt.xlabel('Montecarlo Iterations')
plt.ylabel('Area Under PR Curve')
plt.title('Area Under PR Curve using LightGBM for Different Noise Types')
plt.legend()
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all data frames into one for plotting with Seaborn
combined_df = pd.concat([df.assign(Noise_Type=noise) for df, noise in zip(Area_Under_PR_Curve_dfs, noise_types)])

# Plotting the Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Noise_Type', y='Area Under PR Curve', data=combined_df, palette='Set1')
plt.xlabel('Noise Type')
plt.ylabel('Area Under PR Curve')
plt.title('Area Under PR Curve using LightGBM for Different Noise Types')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine the data frames into a single DataFrame
combined_df = pd.concat([df.rename(columns={'Area Under PR Curve': noise}) for df, noise in zip(Area_Under_PR_Curve_dfs, noise_types)], axis=1)

# Plotting the KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, fill=True, linewidth=2.5)
plt.xlabel('Area Under PR Curve')
plt.ylabel('Density')
plt.title('KDE: Area Under PR Curve of Different Noise Types using LightGBM')
plt.legend(noise_types, loc='upper left')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Concatenate the data frames vertically to form a single data frame
concatenated_df = pd.concat(Area_Under_PR_Curve_dfs, keys=noise_types, names=['Noise Type'])

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=concatenated_df.index.get_level_values('Noise Type'), y=concatenated_df['Area Under PR Curve'])
plt.xlabel('Noise Type')
plt.ylabel('Area Under PR Curve')
plt.title('Area Under PR Curve of Different Noise Types using LightGBM')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Plotting the box plot
plt.figure(figsize=(10, 6))
plt.boxplot([df['Area Under PR Curve'] for df in Area_Under_PR_Curve_dfs], labels=noise_types)
plt.xlabel('Noise Type')
plt.ylabel('Area Under PR Curve')
plt.title('Area Under PR Curve of Different Noise Types using LightGBM')
plt.grid(True)
plt.show()




# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Area Under PR Curve data frames into a single data frame
combined_df = pd.concat(Area_Under_PR_Curve_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Area Under PR Curve', data=combined_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Area Under PR Curve')
plt.title('Area Under PR Curve of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()



# List of noise types
noise_types = ['flicker', 'Gaussian', 'saltnpepper', 'multiplicative', 'colored', 'periodic', '1/f', 'brown', 'uniform', 'impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Area Under PR Curve data frames into a single data frame
combined_df = pd.concat(Area_Under_PR_Curve_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by 'Area Under PR Curve' values
sorted_df = combined_df.sort_values(by='Area Under PR Curve')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Area Under PR Curve', data=sorted_df, palette='Set3')
plt.xlabel('Noise Type')
plt.ylabel('Area Under PR Curve')
plt.title('Area Under PR Curve of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()




# Define a custom color palette for the noise_types
custom_palette = ['blue', 'purple', 'teal', 'brown', 'pink', 'orange', 'green', 'black',  'gray', 'red' ]

# List of noise types
noise_types = ['Flicker', 'Gaussian', 'Salt and Pepper', 'Multiplicative', 'Colored', 'Periodic', '1/f', 'Brown', 'Uniform', 'Impulse']

# List of data frames
Area_Under_PR_Curve_dfs = [flicker_Area_Under_PR_Curve_df, Gaussian_Area_Under_PR_Curve_df, saltnpepper_loss_df, multiplicative_loss_df,
                colored_loss_df, periodic_loss_df, onebyf_loss_df, brown_loss_df, uniform_loss_df, impulse_loss_df]

# Combine all Area Under PR Curve data frames into a single data frame
combined_df = pd.concat(Area_Under_PR_Curve_dfs, keys=noise_types)

# Reset the index for the combined data frame
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Noise Type'}, inplace=True)

# Sort the combined data frame by 'Area Under PR Curve' values
sorted_df = combined_df.sort_values(by='Area Under PR Curve')

# Plotting the box plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Noise Type', y='Area Under PR Curve', data=sorted_df, palette=custom_palette)
plt.xlabel('Noise Type')
plt.ylabel('Area Under PR Curve')
plt.title('Area Under PR Curve of Different Noise Types using LightGBM')
plt.grid(True)
plt.xticks(rotation=45)  # To rotate the x-axis labels for better readability
plt.tight_layout()  # To ensure all elements of the plot fit within the figure area
plt.show()








