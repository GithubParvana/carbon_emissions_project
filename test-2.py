import pandas as pd

# Example DataFrame
data = {
    "Country": ["USA", "China", "India", "Brazil"],
    "CO2_Emissions": [5000, 10000, 3000, 2000],
}
df = pd.DataFrame(data)

print(df)

# Get the count of rows
row_count = df.shape[0]
print(f"The DataFrame has {row_count} rows.")



#   ---   Matploitlib   ---
import matplotlib.pyplot as plt


# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]


# Create a simple line plot
plt.plot(x, y)


# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')


# Display the plot
plt.show()




# Histogram 

# Sample data
data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5]


# Create a histogram
plt.hist(data, bins=5, color='orange', edgecolor='black')


# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')


# Display the plot
plt.show()










# Visualization

# Load the dataset from csv
data = pd.read_csv('formatted_file.csv')


# leave only the columns we need
data = data[['Country', 'Region', 'Date', 'Kilotons of Co2', 'Metric Tons Per Capita']]


# # Create a histogram
plt.hist(data['Country'], bins=25, edgecolor='white')



# Change the x-ticks
xticks = range(0, 400, 10)
plt.xticks(xticks)

# Set a title
plt.title('Histogram')


# Display the plot
plt.show()