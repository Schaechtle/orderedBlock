orderedBlock
============

Re-impmlementation of "Structured Priors for Structure Learning". Now the code is restructured to use a compositional language for transition operators, in the style of BLAISE [Bonawitz, 2008] using generic Kernel, State and Density classes.
---------------------------------

I use a Gibbs sampling throughout. Furthermore, I introduce a periodic re-initialisation to improve results. The current version uses 200 data points, 2000 MCMC moves, 10 ten runs for the selective model averaging. 

---------------------------------
Simply pypy runExperiment.py. You can define which prior you want to use with the -p option:
pypy runExperiment.py -p block
pypy runExperiment.py -p orderedBlock
pypy runExperiment.py -p sparse
pypy runExperiment.py -p uniform (default)
By default 2000 mcmc steps are taken and the a periodic restart is in use after a period of 100 steps. This can be changed with the -s (int), -rs (bool) and -rsp (int) options. For the sparse and uniform priors this is highly recommended since more steps are needed to get a decent result. For example, use a uniform prior without a periodic restart with 50000 mcmc steps:
pypy runExperiment.py -p uniform  -rs False -s 50000
---------------------------------



Issues:
I use quite a lot of deepcopies which seem super inefficient. Furthermore, there can be isssues with identifiability.
