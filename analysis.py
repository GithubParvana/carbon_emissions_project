import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import reduce

from tabulate import tabulate




#  ---      Opening .xlsx file in VS Code using Python (Pandas Library)     ----
import pandas as pd



# Load the Excel file
input_file = "datasets/Carbon_(CO2)_Emissions_by_Country.xlsx"
output_file = "formatted_file.csv"


# Read the Excel file
df = pd.read_excel(input_file)
print(df)


# Save it as a CSV
df.to_csv(output_file, index=False)

print(f"File successfully converted to {output_file}")





#    ---   Tabular format   ---  
dataset = pd.read_csv(output_file, header=1, names=['Country', 'Region', 'Date', 'Kilotons of Co2', 'Metric Tons Per Capita'])


# Column headers  - manually using ->headers
# headers = ['Country', 'Region', 'Date', 'Kilotons of Co2', 'Metric Tons Per Capita']


# Get the headers (column names)
headers_1 = df.columns.tolist()
print(headers_1)


# Create the table
table = tabulate(dataset, headers=headers_1, tablefmt="grid")

# print(table)

file = open('test_file.txt', 'w', encoding='utf-8')
file.write(table)




# ---     Some operations on Dataset - Loading Data    ---   

# Get a count of rows
print(len(df))   # 5677


# Get the size (rows, columns)
# print(df.shape)
print(f"The number of rows is: {df.shape[0]}\nThe number of columns is: {df.shape[1]}")


# Get data types of the columns
data_types = df.dtypes
print(data_types)




# ---   Dataset Inspection and Understanding   ---

""" 
What is the range of years or time period covered in the data?
"""

# Find the range of years
from datetime import datetime as dt


df['Date'] = df['Date'].dt.strftime('%Y')   # Formatting the 'Date' column

start_year = df["Date"].min()
end_year = df["Date"].max()

# print(start_year)
# print(end_year)

print(f"The dataset covers the time period from {start_year} to {end_year}")



# Group the dataset by years to see how many records exist for each year
year_counts = df["Date"].value_counts().sort_values()
print(year_counts)


# Reorder the dataset by "Year" from min to max
df_sorted = df.sort_values(by="Date")
df_sorted.reset_index(drop=True, inplace=True)
print(df_sorted)



# Convert to DataFrame
data_frame = pd.DataFrame(df)


# Aggregate emissions by region to find which contributes most to global emissions
region_emissions = df.groupby("Region")["Kilotons of Co2"].sum().reset_index()


# Save aggregated data to a text file with column headers
with open("region_emissions.txt", "w", encoding='utf-8') as outfile:
    outfile.write(region_emissions.to_string(index=False, header=True))
    

# Create a file to save Aggregated data, then ....
# Save it as a CSV
output_csv_data = "output.csv"


#    ---   Tabular format   ---  
dataset_1 = pd.read_csv(output_csv_data)


# Column headers  - manually using -> headers
headers_2 = ['Region', 'Kilotons of Co2']


# Create the table
table_1 = tabulate(dataset_1, headers=headers_2, tablefmt="grid")

# print(table_1)


# Open and Write to a file
file = open('region_emissions_tabular.txt', 'w', encoding='utf-8')
file.write(table_1)
file.close()


print()
# print("999999999999999999999999")

""" 
Code focues on the "Asia" region, 
sorts countries within it by CO2 emissions in descending order, and extracts the top emitters.

This is useful for identifying the most significant contributors to CO2 emissions within a particular region.
"""
# Analyze top 5 countries within the largest-emitting region
top_countries = df[df["Region"] == "Asia"].sort_values(by="Kilotons of Co2", ascending=False)
print(top_countries.head())


with open("top_countries.csv", "w", encoding='utf-8') as top_country:
    top_country.write(top_countries.head().to_string(index=False, header=True))



# Create a file to save Aggregated data, then ....
# Save it as a CSV
top_country_container = "./top_countries.csv"


#   ---   Tabular format   ---  
dataset_2 = pd.read_csv(top_country_container)


# Get the headers (column names)
headers_3 = ["Region", "Country", "Kilotons of Co2"]


# # Create the table
table_2 = tabulate(dataset_2, headers=headers_3, tablefmt="grid")
print(table_2)


# Open and Write to a file
file_3 = open('top_countries_by_region.txt', 'w', encoding='utf-8')
file_3.write(table_2)
file_3.close()






# ---   Questions  --- 

"""
1) What are the trends in CO2 emissions over the years?
"""

# To analyze the trends in CO2 emissions over the years in Python


"""
Load your dataset into a pandas DataFrame:

"""

dataset_3 = pd.read_csv("formatted_file.csv")

dataset_3['Date'] = pd.to_datetime(dataset_3['Date'])

dataset_3["Year"] = dataset_3["Date"].dt.year


yearly_emissions = dataset_3.groupby('Year')["Kilotons of Co2"].sum().reset_index()

print(yearly_emissions)



# Visualize Trends
plt.figure(figsize=(10, 6))
plt.plot(yearly_emissions["Year"], yearly_emissions["Kilotons of Co2"], marker="o", color="b", label="CO2 Emissions")
plt.title("Trends in CO2 Emissions Over the Years")
plt.xlabel("Year")
plt.ylabel("Kilotons of CO2")
# plt.grid(True)
# plt.legend()
# plt.show()

# Example: Annotate the year with the maximum CO2 emissions

# Find the year with the maximum CO2 emissions
max_year = yearly_emissions.loc[yearly_emissions["Kilotons of Co2"].idxmax()]


# Annotate the maximum point
plt.annotate(f"Max: {max_year['Kilotons of Co2']} in {max_year['Year']}",
             xy=(max_year["Year"], max_year["Kilotons of Co2"]),
             xytext=(max_year["Year"] + 1, max_year["Kilotons of Co2"] + 500),
             arrowprops=dict(facecolor='red', arrowstyle='->'))

# Add grid and legend
plt.grid(True)
plt.legend()
plt.show()






""" 
To find the country in Asia with the highest CO2 emissions over time

2) Which country in Asia has the highest CO2 emissions over time?

"""

# Filter the dataset for the Asia region
asia_data = df[df["Region"] == "Asia"]


# Aggregate Emissions by Country
# Group by country and calculate the total CO2 emissions
asia_emissions = asia_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()


# Sort the countries by total CO2 emissions in descending order
asia_emissions = asia_emissions.sort_values(by="Kilotons of Co2", ascending=False)


# Format 'Kilotons of Co2' column for display
# asia_emissions["Kilotons of Co2"] = asia_emissions["Kilotons of Co2"].map('{:,.2f}'.format)


"""
Keep a Separate Display Column:

Create a new column for the formatted strings while keeping the original column numeric:

'Kilotons of Co2' remains numeric for further calculations.

'Formatted Co2' is used only for displaying.

"""
# Create a formatted display column
asia_emissions["Formatted Co2"] = asia_emissions["Kilotons of Co2"].map('{:,.2f}'.format)


# Display the (DataFrame) - the top emitters
print(asia_emissions[["Country", "Kilotons of Co2", "Formatted Co2"]].head(10))
# print(asia_emissions[["Country", "Kilotons of Co2", "Formatted Co2"]])  - all countries


# Display the top emitters
# print(asia_emissions.head(10))


# Check the type and contents of the Kilotons of Co2 column:
print(asia_emissions["Kilotons of Co2"].dtype)


with open("top_emissions_in_asia.txt", "w", encoding='utf-8') as emission_asia_file:
    emission_asia_file.write(asia_emissions[["Country", "Kilotons of Co2", "Formatted Co2"]].head(10).to_string(index=False, header=True))
    emission_asia_file.close()



# Visualize the Top Emittiers  - Using Matplotlib

# Select the top 10 emittiers
top_emitters = asia_emissions.head(10)


# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_emitters["Country"], top_emitters["Kilotons of Co2"], color="skyblue")
plt.title("Top 10 CO2 Emitters in Asia")
plt.xlabel("Country")
plt.ylabel("Total CO2 Emissions (Kilotons)")
plt.xticks(rotation=45)
plt.show()



highest_emitter = asia_emissions.iloc[0]

print(f"The country with the higest CO2 emissions is {highest_emitter['Country']} with {highest_emitter['Kilotons of Co2']} kilotons.")





# Together -----
# asia_data = df[df["Region"] == "Asia"]
africa_data = df[df["Region"] == "Africa"]
americas_data = df[df["Region"] == "Americas"]
oceania_data = df[df["Region"] == "Oceania"]
europe_data = df[df["Region"] == "Europe"]



while True:
    # if africa_data:
    africa_emissions = africa_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

    africa_emissions = africa_emissions.sort_values(by="Kilotons of Co2", ascending=False)

    africa_emissions["Formatted Co2_Africa"] = africa_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

    print(africa_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Africa"]].head(10))


    with open("top_emissions_in_africa.txt", "w", encoding='utf-8') as emission_africa_file:
        emission_africa_file.write(africa_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Africa"]].head(10).to_string(index=False, header=True))
        emission_africa_file.close()


    top_emitters_africa = africa_emissions.head(10)


    plt.figure(figsize=(12, 6))
    plt.bar(top_emitters_africa["Country"], top_emitters_africa["Kilotons of Co2"], color="magenta")
    plt.title("Top 10 CO2 Emitters in Africa")
    plt.xlabel("Country")
    plt.ylabel("Total CO2 Emissions (Kilotons)")
    plt.xticks(rotation=45)
    plt.show()

    highest_emitter_africa = africa_emissions.iloc[0]

    print(f"The country with the higest CO2 emissions is {highest_emitter_africa['Country']} with {highest_emitter_africa['Kilotons of Co2']} kilotons.")


# elif americas_data:
    americas_emissions = americas_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

    americas_emissions = americas_emissions.sort_values(by="Kilotons of Co2", ascending=False)

    americas_emissions["Formatted Co2_Americas"] = americas_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

    print(americas_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Americas"]].head(10))


    with open("top_emissions_in_americas.txt", "w", encoding='utf-8') as emission_americas_file:
        emission_americas_file.write(americas_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Americas"]].head(10).to_string(index=False, header=True))
        emission_americas_file.close()


    top_emitters_americas = americas_emissions.head(10)


    plt.figure(figsize=(12, 6))
    plt.bar(top_emitters_americas["Country"], top_emitters_americas["Kilotons of Co2"], color="red")
    plt.title("Top 10 CO2 Emitters in Americas")
    plt.xlabel("Country")
    plt.ylabel("Total CO2 Emissions (Kilotons)")
    plt.xticks(rotation=45)
    plt.show()

    highest_emitter_americas = americas_emissions.iloc[0]

    print(f"The country with the higest CO2 emissions is {highest_emitter_americas['Country']} with {highest_emitter_americas['Kilotons of Co2']} kilotons.")


# elif oceania_data:
    oceania_emissions = oceania_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

    oceania_emissions = oceania_emissions.sort_values(by="Kilotons of Co2", ascending=False)

    oceania_emissions["Formatted Co2_Oceania"] = oceania_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

    print(oceania_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Oceania"]].head(10))


    with open("top_emissions_in_oceania.txt", "w", encoding='utf-8') as emission_oceania_file:
        emission_oceania_file.write(oceania_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Oceania"]].head(10).to_string(index=False, header=True))
        emission_oceania_file.close()


    top_emitters_oceania = oceania_emissions.head(10)


    plt.figure(figsize=(12, 6))
    plt.bar(top_emitters_oceania["Country"], top_emitters_oceania["Kilotons of Co2"], color="green")
    plt.title("Top 10 CO2 Emitters in Oceaina")
    plt.xlabel("Country")
    plt.ylabel("Total CO2 Emissions (Kilotons)")
    plt.xticks(rotation=45)
    plt.show()

    highest_emitter_oceania = oceania_emissions.iloc[0]

    print(f"The country with the higest CO2 emissions is {highest_emitter_oceania['Country']} with {highest_emitter_oceania['Kilotons of Co2']} kilotons.")


# elif europe_data:
    europe_emissions = europe_data.groupby("Country")["Kilotons of Co2"].sum().reset_index()

    europe_emissions = europe_emissions.sort_values(by="Kilotons of Co2", ascending=False)

    europe_emissions["Formatted Co2_Europe"] = europe_emissions["Kilotons of Co2"].map('{:,.2f}'.format)

    print(europe_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Europe"]].head(10))

    top_emitters_europe = europe_emissions.head(10)


    with open("top_emissions_in_europe.txt", "w", encoding='utf-8') as emission_europe_file:
        emission_europe_file.write(europe_emissions[["Country", "Kilotons of Co2", "Formatted Co2_Europe"]].head(10).to_string(index=False, header=True))
        emission_europe_file.close()



    plt.figure(figsize=(12, 6))
    plt.bar(top_emitters_europe["Country"], top_emitters_europe["Kilotons of Co2"], color="orange")
    plt.title("Top 10 CO2 Emitters in Europe")
    plt.xlabel("Country")
    plt.ylabel("Total CO2 Emissions (Kilotons)")
    plt.xticks(rotation=45)
    plt.show()

    highest_emitter_europe = europe_emissions.iloc[0]

    print(f"The country with the higest CO2 emissions is {highest_emitter_europe['Country']} with {highest_emitter_europe['Kilotons of Co2']} kilotons.")

    break




# Calculate Region's Total Emissions:
"""
Sum all CO2 emissions for Asia

"""

# Calculate Asia's total emissions
asia_total_emissions = asia_emissions["Kilotons of Co2"].sum()

# Select the top emitters (for example, top 10 emitters in Asia)
top_emitters = asia_emissions.head(10).copy()

# Calculate percentage contribution for each top emitter
top_emitters["Percentage Contribution"] = (top_emitters["Kilotons of Co2"] / asia_total_emissions) * 100
top_emitters["Percentage Contribution"] = top_emitters["Percentage Contribution"].round(2)


"""
This approach keeps the "Percentage Contribution" column
numeric internally while showing the % symbol in your output

"""
# Display the result using Pandas Styler (Temporary Formatting for Display)
# print(top_emitters.style.format({"Percentage Contribution": "{:.2f}%"}))
print(top_emitters[["Country", "Kilotons of Co2", "Percentage Contribution"]])


with open("top_emitters_in_asia.txt", "w", encoding='utf-8') as emitter_asia_file:
    emitter_asia_file.write(top_emitters[["Country", "Kilotons of Co2", "Percentage Contribution"]].to_string(index=False, header=True))
    emitter_asia_file.close()


# Visualize the percentage contributions using a pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    top_emitters["Percentage Contribution"],
    labels=top_emitters["Country"],
    autopct='%1.1f%%',
    startangle=140
)
plt.title("Top Emitters' Contribution to Asia's Total Emissions")
plt.show()





# Sum all CO2 emissions for each Regions with %
africa_total_emissions = africa_emissions["Kilotons of Co2"].sum()
americas_total_emissions = americas_emissions["Kilotons of Co2"].sum()
oceania_total_emissions = oceania_emissions["Kilotons of Co2"].sum()
europe_total_emissions = europe_emissions["Kilotons of Co2"].sum()




while True:
    
    # Africa
    top_emitters_africa = africa_emissions.head(10).copy()
    top_emitters_africa["Percentage Contribution Africa"] = (top_emitters_africa["Kilotons of Co2"] / africa_total_emissions) * 100
    top_emitters_africa["Percentage Contribution Africa"] = top_emitters_africa["Percentage Contribution Africa"].round(2)
    print(top_emitters_africa[["Country", "Kilotons of Co2", "Percentage Contribution Africa"]])


    with open("top_emitters_in_asia.txt", "w", encoding='utf-8') as emitter_africa_file:
        emitter_africa_file.write(top_emitters_africa[["Country", "Kilotons of Co2", "Percentage Contribution Africa"]].to_string(index=False, header=True))
        emitter_africa_file.close()


    plt.figure(figsize=(8, 8))
    plt.pie(
        top_emitters_africa["Percentage Contribution Africa"],
        labels=top_emitters_africa["Country"],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Top Emitters' Contribution to Africa's Total Emissions")
    plt.show()
    print()


    # Americas
    top_emitters_americas = americas_emissions.head(10).copy()
    top_emitters_americas["Percentage Contribution (Americas)"] = (top_emitters_americas["Kilotons of Co2"] / americas_total_emissions) * 100
    top_emitters_americas["Percentage Contribution (Americas)"] = top_emitters_americas["Percentage Contribution (Americas)"].round(2)
    print(top_emitters_americas[["Country", "Kilotons of Co2", "Percentage Contribution (Americas)"]])


    with open("top_emitters_in_americas.txt", "w", encoding='utf-8') as emitter_americas_file:
        emitter_americas_file.write(top_emitters_americas[["Country", "Kilotons of Co2", "Percentage Contribution (Americas)"]].to_string(index=False, header=True))
        emitter_americas_file.close()


    plt.figure(figsize=(8, 8))
    plt.pie(
        top_emitters_americas["Percentage Contribution (Americas)"],
        labels=top_emitters_americas["Country"],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Top Emitters' Contribution to Americas's Total Emissions")
    plt.show()
    print()


    # Oceania
    top_emitters_oceania = oceania_emissions.head(10).copy()
    top_emitters_oceania["Percentage Contribution (Oceania)"] = (top_emitters_oceania["Kilotons of Co2"] / oceania_total_emissions) * 100
    top_emitters_oceania["Percentage Contribution"] = top_emitters_oceania["Percentage Contribution (Oceania)"].round(2)
    print(top_emitters_oceania[["Country", "Kilotons of Co2", "Percentage Contribution (Oceania)"]])


    with open("top_emitters_in_oceania.txt", "w", encoding='utf-8') as emitter_oceania_file:
        emitter_oceania_file.write(top_emitters_oceania[["Country", "Kilotons of Co2", "Percentage Contribution (Oceania)"]].to_string(index=False, header=True))
        emitter_oceania_file.close()


    plt.figure(figsize=(8, 8))
    plt.pie(
        top_emitters_oceania["Percentage Contribution (Oceania)"],
        labels=top_emitters_oceania["Country"],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Top Emitters' Contribution to Oceania's Total Emissions")
    plt.show()
    print()


    # Europe
    top_emitters_europe = europe_emissions.head(10).copy()
    top_emitters_europe["Percentage Contribution (Europe)"] = (top_emitters_europe["Kilotons of Co2"] / europe_total_emissions) * 100
    top_emitters_europe["Percentage Contribution (Europe)"] = top_emitters_europe["Percentage Contribution (Europe)"].round(2)
    print(top_emitters_europe[["Country", "Kilotons of Co2", "Percentage Contribution (Europe)"]])


    with open("top_emitters_in_europe.txt", "w", encoding='utf-8') as emitter_europe_file:
        emitter_europe_file.write(top_emitters_europe[["Country", "Kilotons of Co2", "Percentage Contribution (Europe)"]].to_string(index=False, header=True))
        emitter_europe_file.close()


    plt.figure(figsize=(8, 8))
    plt.pie(
        top_emitters_europe["Percentage Contribution (Europe)"],
        labels=top_emitters_europe["Country"],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Top Emitters' Contribution to Europe's Total Emissions")
    plt.show()
    print()

    break



"""
3) Which countries in Asia have the highest and lowest Metric Tons Per Capita?

"""
# Filter data for Asia
# asia_data = df[df["Region"] == "Asia"]

# Calculate the average Metric Tons Per Capita by country
avg_metric_tons = asia_data.groupby("Country")["Metric Tons Per Capita"].mean()
# print(avg_metric_tons)

# Find the highest and lowest emitters
highest_emitter = avg_metric_tons.idxmax()   # Country with highest
lowest_emitter = avg_metric_tons.idxmin()   # Country with lowest


# Create a DataFrame for visualization
comparison_data = avg_metric_tons[[highest_emitter, lowest_emitter]].reset_index()
comparison_data.columns = ["Country", "Average Metric Tons Per Capita"]


# Bar Chart Visualization
plt.figure(figsize=(8, 6))
plt.bar(comparison_data["Country"], comparison_data["Average Metric Tons Per Capita"], color=["red", "blue"])
plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Asia")
plt.ylabel("Average Metric Tons Per Capita")
plt.xlabel("Country")
plt.xticks(rotation=45)
plt.show()


# Print the hishest and lowest countries
print(f"Highest Metric Tons Per Capita in Asia: {highest_emitter} ({avg_metric_tons[highest_emitter]:.2f})")
print(f"Lowest Metric Tons Per Capita in Asia: {lowest_emitter} ({avg_metric_tons[lowest_emitter]:.2f})")

with open("metric_tones_by_asia.txt", "w", encoding='utf-8') as asia_file:
    asia_file.write(f"Highest Metric Tons Per Capita in Asia: {highest_emitter} ({avg_metric_tons[highest_emitter]:.2f})\n")
    asia_file.write(f"Lowest Metric Tons Per Capita in Asia: {lowest_emitter} ({avg_metric_tons[lowest_emitter]:.2f})")
    asia_file.close()



"""
4) Which countries in all Regions have the highest and lowest Metric Tons Per Capita?

"""

while True:

    # Africa
    avg_metric_tons_africa = africa_data.groupby("Country")["Metric Tons Per Capita"].mean()

    highest_emitter_africa = avg_metric_tons_africa.idxmax()   # Country with highest
    lowest_emitter_africa = avg_metric_tons_africa.idxmin()   # Country with lowest

    comparison_data_africa = avg_metric_tons_africa[[highest_emitter_africa, lowest_emitter_africa]].reset_index()
    comparison_data_africa.columns = ["Country", "Average Metric Tons Per Capita"]

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_data_africa["Country"], comparison_data_africa["Average Metric Tons Per Capita"], color=["red", "blue"])
    plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Africa")
    plt.ylabel("Average Metric Tons Per Capita")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Highest Metric Tons Per Capita in Africa: {highest_emitter_africa} ({avg_metric_tons_africa[highest_emitter_africa]:.2f})")
    print(f"Lowest Metric Tons Per Capita in Africa: {lowest_emitter_africa} ({avg_metric_tons_africa[lowest_emitter_africa]:.2f})")

    with open("metric_tons_by_africa", "w", encoding='utf-8') as africa_file:
        africa_file.write(f"Highest Metric Tons Per Capita in Africa: {highest_emitter_africa} ({avg_metric_tons_africa[highest_emitter_africa]:.2f})\n")
        africa_file.write(f"Lowest Metric Tons Per Capita in Africa: {lowest_emitter_africa} ({avg_metric_tons_africa[lowest_emitter_africa]:.2f})")
        africa_file.close()
    print()


    # Americas
    avg_metric_tons_americas = americas_data.groupby("Country")["Metric Tons Per Capita"].mean()

    highest_emitter_americas = avg_metric_tons_americas.idxmax()   # Country with highest
    lowest_emitter_americas = avg_metric_tons_americas.idxmin()   # Country with lowest

    comparison_data_americas = avg_metric_tons_americas[[highest_emitter_americas, lowest_emitter_americas]].reset_index()
    comparison_data_americas.columns = ["Country", "Average Metric Tons Per Capita"]

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_data_americas["Country"], comparison_data_americas["Average Metric Tons Per Capita"], color=["red", "blue"])
    plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Americas")
    plt.ylabel("Average Metric Tons Per Capita")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Highest Metric Tons Per Capita in Americas: {highest_emitter_americas} ({avg_metric_tons_americas[highest_emitter_americas]:.2f})")
    print(f"Lowest Metric Tons Per Capita in Americas: {lowest_emitter_americas} ({avg_metric_tons_americas[lowest_emitter_americas]:.2f})")

    with open("metric_tons_by_americas", "w", encoding='utf-8') as americas_file:
        americas_file.write(f"Highest Metric Tons Per Capita in Americas: {highest_emitter_americas} ({avg_metric_tons_americas[highest_emitter_americas]:.2f})\n")
        americas_file.write(f"Lowest Metric Tons Per Capita in Americas: {lowest_emitter_americas} ({avg_metric_tons_americas[lowest_emitter_americas]:.2f})")
        americas_file.close()
    print()


    # Oceania
    avg_metric_tons_oceania = oceania_data.groupby("Country")["Metric Tons Per Capita"].mean()

    highest_emitter_oceania = avg_metric_tons_oceania.idxmax()   # Country with highest
    lowest_emitter_oceania = avg_metric_tons_oceania.idxmin()   # Country with lowest

    comparison_data_oceania = avg_metric_tons_oceania[[highest_emitter_oceania, lowest_emitter_oceania]].reset_index()
    comparison_data_oceania.columns = ["Country", "Average Metric Tons Per Capita"]

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_data_oceania["Country"], comparison_data_oceania["Average Metric Tons Per Capita"], color=["red", "blue"])
    plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Oceania")
    plt.ylabel("Average Metric Tons Per Capita")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Highest Metric Tons Per Capita in Oceania: {highest_emitter_oceania} ({avg_metric_tons_oceania[highest_emitter_oceania]:.2f})")
    print(f"Lowest Metric Tons Per Capita in Oceania: {lowest_emitter_oceania} ({avg_metric_tons_oceania[lowest_emitter_oceania]:.2f})")
    
    with open("metric_tons_by_oceania.txt", "w", encoding='utf-8') as oceania_file:
        oceania_file.write(f"Highest Metric Tons Per Capita in Oceania: {highest_emitter_oceania} ({avg_metric_tons_oceania[highest_emitter_oceania]:.2f})\n")
        oceania_file.write(f"Lowest Metric Tons Per Capita in Oceania: {lowest_emitter_oceania} ({avg_metric_tons_oceania[lowest_emitter_oceania]:.2f})")
        oceania_file.close()
    print()

    # Europe
    avg_metric_tons_europe = europe_data.groupby("Country")["Metric Tons Per Capita"].mean()

    highest_emitter_europe = avg_metric_tons_europe.idxmax()   # Country with highest
    lowest_emitter_europe = avg_metric_tons_europe.idxmin()   # Country with lowest

    comparison_data_europe = avg_metric_tons_europe[[highest_emitter_europe, lowest_emitter_europe]].reset_index()
    comparison_data_europe.columns = ["Country", "Average Metric Tons Per Capita"]

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_data_europe["Country"], comparison_data_europe["Average Metric Tons Per Capita"], color=["red", "blue"])
    plt.title("Countries with Highest and Lowest Metric Tons Per Capita in Europe")
    plt.ylabel("Average Metric Tons Per Capita")
    plt.xlabel("Country")
    plt.xticks(rotation=45)
    plt.show()

    print(f"Highest Metric Tons Per Capita in Europe: {highest_emitter_europe} ({avg_metric_tons_europe[highest_emitter_europe]:.2f})")
    print(f"Lowest Metric Tons Per Capita in Europe: {lowest_emitter_europe} ({avg_metric_tons_europe[lowest_emitter_europe]:.2f})")

    # Save aggregated data to a text file with column headers
    with open("metric_tons_by_europe.txt", "w", encoding='utf-8') as europe_file:
        europe_file.write(f"Highest Metric Tons Per Capita in Europe: {highest_emitter_europe} ({avg_metric_tons_europe[highest_emitter_europe]:.2f})\n")
        europe_file.write(f"Lowest Metric Tons Per Capita in Europe: {lowest_emitter_europe} ({avg_metric_tons_europe[lowest_emitter_europe]:.2f})")
        europe_file.close()

    break




""" 
5) How do CO2 emissions and per capita emissions compare across decades?
"""

# Ensure 'Date' is converted to integer
df["Year"] = df["Date"].astype(int)


# Create a 'Decade' column
df["Decade"] = (df["Year"] // 10) * 10 


# Aggregate data by decade
decade_data = df.groupby("Decade").agg({
    "Kilotons of Co2": "sum",  # Total CO2 emissions
    "Metric Tons Per Capita": "mean"   # Average per capita emissions
}).reset_index()


# Visualization: CO2 Emissions and Per Capita Emissions over Decades
plt.figure(figsize=(12, 6))


# Plot total CO2 emissions
"""
Visualization:

A bar chart shows the total CO2 emissions by decade.
A line plot highlights trends in per capita emissions over time.
"""

plt.subplot(1, 2, 1)
plt.bar(decade_data["Decade"], decade_data["Kilotons of Co2"], color="skyblue")
plt.title("Total CO2 Emissions by Decade")
plt.xlabel("Decade")
plt.ylabel("Kilotons of Co2")
plt.xticks(decade_data["Decade"], rotation=45)


# Plot per capita emissions
plt.subplot(1, 2, 2)
plt.plot(decade_data["Decade"], decade_data["Metric Tons Per Capita"], marker="o", color="orange")
plt.title("Per Capita CO2 Emissions by Decade")
plt.xlabel("Decade")
plt.ylabel("Metric Tons Per Capita")
plt.grid(True)


# Adjust layout and show the plot
plt.tight_layout()
plt.show()




""" 
6) Is there a correlation between total CO2 emissions and Metric Tons Per Capita?
"""
import seaborn as sns

#Remove rows with missing data in the relevant columns
filtered_data = df.dropna(subset=["Kilotons of Co2", "Metric Tons Per Capita"])


# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(
    data=filtered_data,
    x="Kilotons of Co2",
    y="Metric Tons Per Capita",
    line_kws={"color":"red"},
    scatter_kws={"alpha":0.6}
)


# Titles and labels
plt.title("Correlation Between Total CO2 Emissions and Metric Tons Per Capita")
plt.xlabel("Total CO2 Emissions (Kilotons)")
plt.ylabel("Metric Tons Per Capita")
plt.grid(True)
plt.show()


# Calculate Per Capita correlation coefficient
correlation = filtered_data["Kilotons of Co2"].corr(filtered_data["Metric Tons Per Capita"])
print(f"Correlation Coefficient: {correlation:.2f}")


#  --- End ---
