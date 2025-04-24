import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

#OBJECTIVE 1:
#Bar Chart of Total Square Footage with Mean and Z-Score

data = pd.read_csv('2025_IOLP_BUILDINGS_DATASET.csv')
print(data.head())
total = data.groupby('GSA Region')['Building Rentable Square Feet'].sum()
total = total.sort_values(ascending=False)
plt.figure(figsize=(12, 7)) 
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'lime', 'teal', 'gold']  
plt.bar(total.index, total, color=colors, edgecolor='black', width=0.7)  
plt.title('TOTAL SQUARE FEET BY REGION', fontsize=16, color='darkblue')  
plt.xlabel('REGIONS', fontsize=12, color='purple')  
plt.ylabel('SQUARE FEET', fontsize=12, color='purple') 
plt.xticks(rotation=45, fontsize=10)  
mean = total.mean()  
plt.axhline(mean, color='red', linestyle='--', linewidth=2, label=f'Average: {mean:.2f}')  
plt.legend(fontsize=10) 
plt.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
plt.show()
z = (total - mean) / total.std()
print('Average square feet is:', mean)
print(" ")
print('Z-scores for each region:\n', z)
print(" ")
print('The most extreme region is', z.idxmax(), 'with a z-score of', z.max())
print(" ")

#OBJECTIVE 2:
#Pie Chart  and Box Plot of Property Counts with Covariance
data = pd.read_csv('2025_IOLP_BUILDINGS_DATASET.csv')
count = data['GSA Region'].value_counts()
plt.figure(figsize=(8, 8)) 
plt.pie(count, labels=count.index, autopct='%1.1f%%', colors=['lightblue', 'lightpink', 'lightgreen', 'cyan', 'blue', 'yellow', 'orange', 'violet', 'lime', 'teal', 'gold'])
plt.title('PROPERTIES IN EACH REGION', fontsize=16, color='darkgreen') 
plt.show()
totals = data.groupby('GSA Region')['Building Rentable Square Feet'].sum()
cov = np.cov(totals, count)[0, 1]
print('Covariance between square feet and counts is:', cov)
print('Region with most properties is', count.idxmax(), 'with', count.max())
print('Region with least properties is', count.idxmin(), 'with', count.min())


#OBJECTIVE 3:
#Scatter of Counts vs Square Feet (scatter),Square Feet Box Plot (Box)

data = pd.read_csv('2025_IOLP_BUILDINGS_DATASET.csv')
summary = data.groupby('GSA Region').agg({
    'Building Rentable Square Feet': 'sum',
    'Real Property Asset Name': 'count'
})
summary.columns = ['Square Feet', 'Count']
plt.figure(figsize=(10, 6))  
plt.scatter(summary['Count'], summary['Square Feet'], color='blue', s=100)  
plt.title('SQUARE FEET AND COUNTS', fontsize=12, color='blue') 
plt.xlabel('COUNT OF PROPERTIES', fontsize=10, color = "blue")
plt.ylabel('SQUARE FEET', fontsize=10, color = "blue")
plt.plot(summary.index, summary['Square Feet'], color='red', label='Square Feet Line') 
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.boxplot([data[data['GSA Region'] == r]['Building Rentable Square Feet'] for r in data['GSA Region'].unique()])
plt.title('SQUARE FEET BOX', fontsize=14, color='darkblue')
plt.xlabel('REGIONS', fontsize=12, color = "darkblue")
plt.ylabel('SQUARE FEET', fontsize=12, color = "darkblue")
unique_regions = data['GSA Region'].unique()
plt.xticks(range(1, len(unique_regions) + 1), unique_regions)
plt.grid(True)
plt.show()
cov = np.cov(summary['Square Feet'], summary['Count'])[0, 1]
print(" ")
print('Covariance between square feet and counts is:', cov)



#OBJECTIVE 4
#Average Square Feet per Property
data = pd.read_csv('2025_IOLP_BUILDINGS_DATASET.csv')
avg = data.groupby('GSA Region')['Building Rentable Square Feet'].mean()
avg = avg.sort_values(ascending=False)

# Bar chart
plt.figure(figsize=(10, 6))
plt.bar(avg.index, avg, color='green')
plt.title("AVERAGE SQUARE FEET PER PROPERTY", fontsize=14, color="darkblue")
plt.xlabel('REGIONS', fontsize=12,color="darkblue")
plt.ylabel('AVERAGE SQUARE FEET', fontsize=12,color="darkblue")
plt.xticks(rotation=45) 
plt.grid(True)
plt.show()

#Stacked Line Chart
stats = data.groupby('GSA Region')['Building Rentable Square Feet'].agg(['mean', 'sum'])
plt.figure(figsize=(10, 6))
plt.plot(stats.index, stats['mean'], color='blue', label='Average')
plt.plot(stats.index, stats['sum'], color='red', label='Total') 
plt.title('Stacked Line - Mean and Total', fontsize=14, color='darkblue')
plt.xlabel('REGIONS', fontsize=12,color= "darkblue")
plt.ylabel('SQUARE FEET', fontsize=12,color = "darkblue")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Heatmap
sheet1 = pd.read_csv("2025_IOLP_BUILDINGS_DATASET.csv")
sheet1['Building Rentable Square Feet'] = pd.to_numeric(sheet1['Building Rentable Square Feet'], errors='coerce')
region_data = sheet1.groupby('GSA Region')['Building Rentable Square Feet'].sum().reset_index()
heatmap_data = region_data.pivot_table(
    index='GSA Region',
    values='Building Rentable Square Feet'
)
plt.figure(figsize=(10, 6))
plt.title('Total Rentable Square Footage by GSA Region', fontsize=18, fontweight='bold')
sb.heatmap(
    heatmap_data,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor='white'
)
plt.xlabel("Rentable Square Feet", fontsize=12)
plt.ylabel("GSA Region", fontsize=12)
plt.tight_layout()
plt.show()



#OBJECTIVE 5:
#Geospatial Distribution of Properties(OWNED OR LEASED)

data = pd.read_csv("2025_IOLP_BUILDINGS_DATASET.csv")
geo = data[['Latitude', 'Longitude', 'Owned or Leased', 'City', 'State']].dropna()
print(data.head())
sb.set(style="whitegrid", palette="pastel", font_scale=1.2)
plt.figure(figsize=(14, 8))
plt.title('Geospatial Distribution of Federal Properties\n(Colored by Ownership Type)', fontsize=18, fontweight='bold')

sb.scatterplot(
    data=geo,
    x="Longitude",
    y="Latitude",
    hue="Owned or Leased",
    s=100,
    edgecolor="black",
    alpha=0.7
)

plt.xlabel("LONGITUDE", fontsize=14, color = "darkblue")
plt.ylabel("LATITUDE", fontsize=14,color = "darkblue")
plt.legend(title="OWNERSHIP TYPE", loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

#OBJECTIVE 6:
# Construction Trend Over Time

df = pd.read_csv("2025_IOLP_BUILDINGS_DATASET.csv")
df['Construction Date'] = pd.to_numeric(df['Construction Date'], errors='coerce')
df['Decade'] = (df['Construction Date'] // 10) * 10

# Count constructions per decade
construction_counts = df['Decade'].value_counts().sort_index()

# Plotting
plt.figure(figsize=(10, 6))
palette = sb.color_palette("husl", len(construction_counts))
sb.lineplot(
    x=construction_counts.index,
    y=construction_counts.values,
    marker='o',
    linewidth=2.5,
    markersize=8,
    color='blue'
)
for i, (x, y) in enumerate(zip(construction_counts.index, construction_counts.values)):
    plt.plot(x, y, 'o', markersize=10, color=palette[i])
plt.title("CONSTRUCTION TREND OVER TIME", fontsize=16,color = "darkblue")
plt.xlabel("DECADE", fontsize=12,color="darkblue")
plt.ylabel("NUMBER OF OBSERVATION", fontsize=12,color = "darkblue")
plt.grid(True)
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
