import numpy as np
import matplotlib.pyplot as plt
from functions import csfBestFit
import csv

# Code for preliminary data analysis performed for SPHSC 525 final project

def get_average_CSF(filepath):
    """Pull tested spatial frequencies and average CSF results
    from Results file given a filepath.
    
    Parameters:
        filepath (string): path to TestResults.csv file generated
        by my program.
        
    Returns:
        sfs (numpy array of floats): list of spatial frequencies tested
        average (numpy array of floats): average calculated from all of the
        trials (each row) in the given file."""
    
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file, delimiter=",")
        header = True
        values = []

        for row in csv_reader:
            if header:
                sfs=row
                header = False
            else:
                values.append(row)

    sfs= np.asarray(sfs, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    average = np.mean(values, axis=0)

    return sfs, average



def plot_average_CSF(filepath, subject_number):
    """Plot the average CSF +/- standard deviation and a line of
    best fit  results using data from Results file.
    
    Parameters:
        filepath (string): path to TestResults.csv file generated
        by my program.
        subject_number (int): subject number to use in the plot title
        
    Returns:
        None: Plots results and shows plot, does not return a value"""

    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file, delimiter=",")
        header = True
        values = []

        for row in csv_reader:
            if header:
                sfs=row
                header = False
            else:
                values.append(row)

    sfs= np.asarray(sfs, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    print(f"Average of {values.shape[0]} trials")

    average = np.mean(values, axis=0)
    stdev = np.std(values, axis=0)

    xvals = np.geomspace(0.25,64, 100)
    bestFit = csfBestFit(xvals, sfs, average)

    plt.figure(figsize=(6.5,6))

    plt.scatter(sfs, average, marker = 's', color="k", label="Data +/- StDev")
    plt.errorbar(sfs, average, stdev, ecolor="k", linestyle='')
    plt.plot(xvals, bestFit, linestyle="-", linewidth=2, color='r', label="Best Fit")

    plt.grid(True)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1,200])
    plt.xlim([0.1, 50])
    plt.yticks([1, 10, 50, 100, 200], ['1', '10', '50', '100', '200'])
    plt.xticks([0.1, 0.5, 1, 2, 4, 8, 16, 32], ['0.1', '0.5', '1', '2', '4', '8', '16', '32'])
    plt.xlabel("Spatial Frequency (c/deg)")
    plt.ylabel("Contrast Sensitivity")
    plt.title(f"Subject {subject_number} Results")
    plt.legend()
    plt.show()



def plot_multiple_CSFs(filepaths):
    """Plots multiple average CSFs given a list of filepaths to  CSV
    files containing test results. Data are plotted as square markers
    +/- standard deviation, and best fit lines are computed for each
    subject's data using asymmetric parabolic function.
    
    Parameters:
        filepaths (list of strings): a list of filepaths to multiple CSV
        Results files.
        
    Returns:
        None: Does not return a value. Plots multiple average CSFs in
        different colors and shows that plot"""
    
    sfs = []
    color_options = [[1.0, 0.0, 0.0],
                  [0.0, 0.75, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 1.0, 1.0],
                  [1.0, 0.0, 1.0]]

    plt.figure(figsize=(6.5,6))

    for i in range(len(filepaths)):
        with open(filepaths[i], 'r') as file:
            csv_reader = csv.reader(file, delimiter=",")
            header = True
            vals = []

            for row in csv_reader:
                if header:
                    sfs=row
                    header = False
                else:
                    vals.append(row)
        
        sfs = np.asarray(sfs, dtype=np.float32)
        vals = np.asarray(vals, dtype=np.float32)
        average = np.mean(vals, axis=0)
        stdev = np.std(vals, axis=0)

        xvals = np.geomspace(0.25, 64, 100)
        bestFit = csfBestFit(xvals, sfs, average)
        plt.scatter(sfs, average, marker = 's', color=color_options[i])
        plt.errorbar(sfs, average, stdev, ecolor=color_options[i], linestyle='')
        plt.plot(xvals, bestFit, linestyle="-", linewidth=2, color=color_options[i], label=f"S{i+1}")


    plt.grid(True)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1,200])
    plt.xlim([0.1, 50])
    plt.yticks([1, 10, 50, 100, 200], ['1', '10', '50', '100', '200'])
    plt.xticks([0.1, 0.5, 1, 2, 4, 8, 16, 32], ['0.1', '0.5', '1', '2', '4', '8', '16', '32'])
    plt.xlabel("Spatial Frequency (c/deg)")
    plt.ylabel("Contrast Sensitivity")
    plt.title(f"Individual Results")
    plt.legend()
    plt.show()



def population_csf(filepaths):
    """Compute average CSF using multiple individual subjects' results given
    as a list of filepaths to different results CSV files.
    
    Paramters:
        filepaths (list of strings): List of filepaths to different subjects'
        results CSV files.
        
    Returns:
        None: Does not return a value, computes the mean of all of the
        included subjects' averages, as well as the standard deviation,
        and then plots and shows the results."""
    
    sfs = []
    values = []

    for i in range(len(filepaths)):
        sfs, vals = get_average_CSF(filepaths[i])
        values.append(vals)

    average = np.mean(values, axis=0)
    stdev = np.std(values, axis=0)

    xvals = np.geomspace(0.25, 64, 100)
    bestFit = csfBestFit(xvals, sfs, average)

    plt.figure(figsize=(6.5,6))

    plt.scatter(sfs, average, marker = 's', color="k", label="Data +/- StDev")
    plt.errorbar(sfs, average, stdev, ecolor="k", linestyle='')
    plt.plot(xvals, bestFit, linestyle="-", linewidth=2, color='r', label="Best Fit")

    plt.grid(True)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1,200])
    plt.xlim([0.1, 50])
    plt.yticks([1, 10, 50, 100, 200], ['1', '10', '50', '100', '200'])
    plt.xticks([0.1, 0.5, 1, 2, 4, 8, 16, 32], ['0.1', '0.5', '1', '2', '4', '8', '16', '32'])
    plt.xlabel("Spatial Frequency (c/deg)")
    plt.ylabel("Contrast Sensitivity")
    plt.title(f"Population Results")
    plt.legend()
    plt.show()



def get_peak_frequency(filepath):
    """Gets peak sensitivities of every trial from a given subject by 
    plotting a line of best fit using an asymmetric parabolic function
    and pulling the spatial frequency that corresponds to that peak
    contrast sensitivity.
    
    Parameters:
        filepath (string): A file path that points to a TestResults.csv file
        generated by the CSF testing program.
        
    Returns:
        max_vals (list of floats): a list that includes the peak frequency for
        every trial (each row) included in the TestResults.csv file."""
    
    sfs = []
    max_vals = []
    xvals = np.geomspace(0.25, 64, 100)

    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file, delimiter=",")
        header = True
        values = []

        for row in csv_reader:
            if header:
                sfs=row
                header = False
            else:
                values.append(row)

    sfs= np.asarray(sfs, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    for i in range(values.shape[0]):
        bestFit = csfBestFit(xvals, sfs, values[i,:])
        m = max(bestFit)
        idx = bestFit.index(m)
        max_vals.append(xvals[idx])

    return max_vals
    


def plot_peak_frequency(filepaths):
    """Calculates the average peak spatial frequency for a set of subjects
    and plots the results as a box and whisker plot.
    
    Parameters:
        filepaths (list of strings): a list of filepaths pointing to multiple
        TestResults.csv files generated by the CSF testing program.
        
    Returns:
        None: does not return a value. Shows box and whisker plot of peak
        frequencies across subjects"""
    
    sfs = []
    max_vals=[]
    xvals = np.geomspace(0.25, 64, 100)
    
    for i in range(len(filepaths)):
        sfs, vals = get_average_CSF(filepaths[i])
        bestFit = csfBestFit(xvals, sfs, vals)
        m = max(bestFit)
        idx = bestFit.index(m)
        max_vals.append(xvals[idx])

    plt.figure(figsize=(6.5,6))
    plt.boxplot(max_vals)
    plt.title("SF at Peak Sensitivitiy")
    plt.ylim([2,3])
    plt.ylabel("Spatial Frequency")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.show()



def plot_peak_sensitivity(filepaths):
    """Calculates the average peak contrast sensitivity for multiple subjects
    and plots the population results as a box and whisker plot.
    
    Parameters:
        filepaths (list of strings): a list of filepaths pointing to multiple
        TestResults.csv files generated by the CSF testing program.
        
    Returns:
        None: does not return a value. Shows box and whisker plot of peak
        contrast sensitivity across subjects"""
    
    sfs = []
    max_vals=[]
    xvals = np.geomspace(0.25, 64, 100)
    
    for i in range(len(filepaths)):
        sfs, vals = get_average_CSF(filepaths[i])
        bestFit = csfBestFit(xvals, sfs, vals)
        max_vals.append(max(bestFit))

    plt.figure(figsize=(6.5,6))
    plt.boxplot(max_vals)
    plt.title("Peak Sensitivitiy")
    plt.ylim([120, 150])
    plt.ylabel("Spatial Frequency")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.show()



def plot_cutoff_frequency(filepaths):
    """Calculates the average cutoff frequency for multiple subjects
    and plots the population results as a box and whisker plot.
    
    Parameters:
        filepaths (list of strings): a list of filepaths pointing to multiple
        TestResults.csv files generated by the CSF testing program.
        
    Returns:
        None: does not return a value. Shows box and whisker plot of cutoff
        frequency across subjects"""
    
    sfs = []
    cutoff_vals=[]
    xvals = np.geomspace(0.25, 64, 100)
    
    for i in range(len(filepaths)):
        sfs, vals = get_average_CSF(filepaths[i])
        bestFit = csfBestFit(xvals, sfs, vals)
        m = min(bestFit, key= lambda x:abs(x-1))
        idx = bestFit.index(m)
        cutoff_vals.append(xvals[idx])

    plt.figure(figsize=(6.5,6))
    plt.boxplot(cutoff_vals)
    plt.ylim([38,42])
    plt.title("Cutoff Frequency")
    plt.ylabel("Spatial Frequency")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.show()