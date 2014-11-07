orderedBlock
============


The main code can be found in orderedBlock and orderedBlock_STRICT. I realised that if I interpret the word order as in eq. 3 of the "Structured Priors for Structure Learning" paper, I will get results as depicted in Fig. 2 (attached). This makes sense, because my score is purely Bayesian and does not penalise more complicated structures. However, if I interpret the word ordering more strictly (i.e. replace ob > oa with ob == oa+1), I get the correct result (Fig. 3, much more like the real thing but still way more noisy than described in the paper). 

---------------------------------


Simply run pypy orderBlock_selectiveModelav.py (orderBlock_selectiveModelav_STRICT.py resp.)  to run the algorithm. Run python plotOutput.py  to reproduce the figures (I was too lazy to find a way to plot in pypy...).

---------------------------------

I use a Metropolis-Hastings-like algorithm for the edges and Gibbs for the classes and orders. Furthermore, I introduce a periodic re-initialisation to improve results. The current version uses 200 data points, 2000 MCMC moves, 10 ten runs for the selective model averaging.
