import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


#randomly sample from a standard normal distibution
samp = stats.norm.rvs(size=250)


#create a frequency distibution to test against
hist = plt.hist(samp, bins = 10)


#create percentiles using a cdf of the created distribution
percentiles = np.cumsum(hist[0])/len(samp)


bins = hist[1][1:]

#create a list of distributions to test against
dist = ['norm', 'expon', 'logistic']

#list to store probability values for chi squared distribution 
chi_p = []
#list to store cdf probabilities for probability plot 
test_perc = []
degrees = len(hist[1]) - 1



for x in dist:

    attr = getattr(stats, x)

    #estimate paramters for each hypothosized disitbution
    test = stats.fit(attr, samp)

    #compare bins of testing distribtuion against CDF of hypothosized distirbution 
    test_percentiles = attr.cdf(list(bins), loc=test.params[0], scale = test.params[1])

    test_perc.append(test_percentiles)

    #calculate chi-square statstic manually
    chi2_stat = np.sum((np.square(test_percentiles - percentiles))/percentiles)

    #use pdf of chi squared distribtution to find probability value associate with the statstic 
    chi_p.append(stats.chi2.cdf(chi2_stat, df=degrees))


print("The best fit disitbution: ", dist[chi_p.index(np.min(chi_p))])
print("chi squared stat: ", np.min(chi_p))
print("p-value: ", stats.chi2.pdf(np.max(chi_p), df=degrees))


print('\n\n', test_perc)



#create PP probability plots to visualize test results
plt.clf()

sns.set_style('dark')
sns.despine()
sns.set_context('paper')


plt.subplot(131)
sns.lineplot(x=np.arange(0, 1, .1), y=np.arange(0, 1, .1))
sns.scatterplot(x=test_perc[0], y=percentiles)
plt.xlabel("Theoretical CDF")
plt.ylabel("Empirical CDF")
plt.title(dist[0])


plt.subplot(132)
sns.lineplot(x=np.arange(0, 1, .1), y=np.arange(0, 1, .1))
sns.scatterplot(x=test_perc[1], y=percentiles)
plt.xlabel("Theoretical CDF")
plt.ylabel("Empirical CDF")
plt.title(dist[1])


plt.subplot(133)
sns.lineplot(x=np.arange(0, 1, .1), y=np.arange(0, 1, .1))
sns.scatterplot(x=test_perc[2], y=percentiles)
plt.xlabel("Theoretical CDF")
plt.ylabel("Empirical CDF")
plt.title(dist[2])

plt.subplots_adjust(hspace = 2)
plt.show()