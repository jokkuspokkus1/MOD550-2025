import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

class dataaquisition:

   
    
    def __init__(self,n=1000):
        self.n=n
        self.data1=[]
        self.data2=[]

    def generatedata(self):
        for i in range(self.n):
            self.data1.append(random.gauss(100,15))
            self.data2.append(random.gauss(100,15))

    def getdata(self):
        return self.data1,self.data2
    
    def plot_heatmap(self):
        plt.hist2d(self.data1,self.data2,bins=30)
        plt.colorbar()
        plt.xlabel('data1')
        plt.ylabel('data2')
        plt.title('Heatmap of data1 and data2')
        plt.show()

    def histogram_from_file(self, filename):
        df = pd.read_csv(filename)   
        values = df["height_cm"].to_numpy() 

        bins = 10
        counts, edges = np.histogram(values, bins=bins)

        plt.hist(values, bins=bins)
        plt.xlabel('Value'); plt.ylabel('Frequency')
        plt.title('Histogram of all values in file')
        plt.show()

        return counts, edges  
    
    def pmf_from_counts(self, counts):
        total = sum(counts)
        if total == 0:
            return np.zeros_like(counts, dtype=float)
        return counts / total
    
    def plot_pmf(self, pmf, title="PMF"):
        plt.bar(range(len(pmf)), pmf)
        plt.xlabel("bin index")
        plt.ylabel("Probability")
        plt.title(title)
        plt.show()

    def cumulative_from_pmf(self, pmf):
        cs=np.cumsum(pmf)
        plt.plot(cs)
        plt.xlabel("bin index")
        plt.ylabel("Cumulative Probability")
        plt.title("Cumulative Distribution Function")
        plt.show()
        return cs
