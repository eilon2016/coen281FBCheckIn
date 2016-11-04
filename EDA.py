# Exploratory Data Analysis for the FB Check In Training Set
import numpy
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import pandas
import os

"""
VARIABLE DESCRIPTIONS
Row_id          Unique Row_id           [0, 29,118,020]
x               X coordinate            [0, 10]
y               Y coordinate            [0, 10]
accuracy        Measure of accuracy     [1, 1033]
time            Time of check in        [1, 786239]
place_id        Check in location
"""

def loadData(dir, fileName):
    dataPath = dir + "/" + fileName + ".csv"
    picklePath = dir + "/" + fileName + ".pkl"
    # Using pickle allows for faster subsequent runs
    if not os.path.isfile(picklePath):
        data = pandas.read_csv(dataPath)
        data.to_pickle(picklePath)
    else:
        data = pandas.read_pickle(picklePath)
    return data

def main():
    f = open('output.txt', 'w') # Output file
    pandas.set_option('display.float_format', lambda x: '%.3f' % x) # up to 3 decimal places
    matplotlib.interactive(False)

    # Load our data
    data = loadData("/Users/Johnny/Documents/FBData", "train")

    # Get initial description of data: count, mean, std, min, quartiles, max
    s = data.describe().to_string()
    print(s)
    f.write(s)
    # Get counts for how many check ins a place_id has
    place_id = data['place_id'] # Place_id is our class/label; it contains a wide range of counts
    place_id_freq = place_id.value_counts()
    print("     place_id_freq")
    f.write("\n     place_id_freq\n")
    s = place_id_freq.describe().to_string()
    print(s)
    f.write(s)

    # Generates a histogram: x axis is the number of check-ins, y is how many place_ids have that number of check-ins
    plt.figure() # Make a new figure
    plt.hist(place_id_freq.values, bins='fd') # Histogram of the counts
    plt.xticks(numpy.arange(0, 1900, 100))
    plt.xlabel("Number of Check-ins")
    plt.ylabel("Number of place_ids")
    plt.savefig('Figures/Check_in_count.png')



    f.close() # Close output.txt

if __name__ == "__main__":
    main()
