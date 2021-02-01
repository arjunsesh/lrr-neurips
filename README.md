# Learning Rich Rankings

Jan 31, 2020

This repository contains the code used for the paper [Learning Rich Rankings](https://arjunsesh.github.io/files/lrr_neurips.pdf). The repository uses PyTorch to learn probabilistic models for ranking data by learning on discrete choice data made from rankings.

Data
====
Data from different domain areas used in the paper appear as subdirectories of the "data" folder. For example, each of the election datasets appears in /data/election, which contains the files dublin-north.soi, dublin-west.soi, and meath.soi. Each of these files contains the ranked choice votes affiliated from an election.

The .soi file extension and format was adapted from the data collection  [preflib](http://www.preflib.org/), and many of the datasets appearing in this repo came from their website. Others, such as the NASCAR dataset, were transformed into preflib data before uploading. Our repository also includes .soc files, another preflib data format indicating a dataset consisting of complete rankings (no partial rankings).

Example: Fitting and Evaluating Ranking Models on Real Data
====
To fit a suite of repeated selection models to a dataset from LRR's Table 1, we simply run src/train.py from the top directory of the repo and use flags to stipulate the dataset and alter some of the training options. For example, to fit to the LETOR collection of datasets, we would run the following:

```python
python src/train.py -dset=letor -num_dsets=10
```

This command will automatically loop over the directory and fit 4 models, CRS with r=1,4 and 8, and a Plackett-Luce model, to train sets for 5-fold CV to a random num_dsets=10 datasets in the data/letor directory which meet the other default requirements, e.g. a maximum of 1000 rankings. We have capped the number of datasets to 10 in this example so that the training is expedient- training 5 models on 5 folds means that we are training 250 models here, using a default of 100 epochs for each.

Across different collections of datasets in the data folder there is significant heterogeneity in the number of alternatives ranked, number of rankings, lengths (mostly partial vs. mostly complete) and number of datasets each collection, so the default values often need to be tweaked depending on the data we want to fit.

The learned model parameters and all of the folds for CV will be stored in pickled dictionaries in /cache/learned_models/letor/
Each file in that directory will correspond to one of the LETOR datasets. To compute the out-of-sample errors for all of these models, we simply need to run

```python
python src/test.py -dset=letor
```

This will output pickled dictionaries of the out of sample log-loss of the RS choices given by each ranking to cache/computed_errors, which can then be visualized via:

```python
python src/plot.py -dset=letor
```

This will output pdfs for each letor dataset we considered, plotting the average log loss of predicting the next entry as a function of the positions in the ranking. Because many of the letor datasets are small, we may want to include a histogram telling us how many rankings have each position, which we can do by adding a flag:

```python
python src/plot.py -dset=letor -hist=y
```

We've used the letor datasets so far because they are very small and thus allow for fast training and testing, but the histograms and plots will show that many of these datasets are too small to have interesting output, and comparing the performance across many datasets of various sizes can be difficult. If we want to look at simple averages of out-of-sample log-likelihood, we can compute and print these to a text file via:

```python
python src/test.py -dset=letor -setsize=n
python src/plot.py -dset=letor -setsize=n -all=y
```

And look at plots/letor/soi-Llog-all.txt

Notes
==
As discussed in the paper, the Mallows model cannot be implemented with Pytorch as with the other repeated selection models because it has a discrete parameter space, meaning we cannot try to compute the MLE using gradient descent based models. However, we are able to approximate the MLE the Mallows model with src/mallows-approx.py.

The implementation of the mallows-approximation is designed for ease of model comparison, and should be run after training other models, as it grabs data from the cached folds stored for cross validation. We simply run

```python
python src/mallows-approx.py -dset=letor
```

and the mean out-of-sample log-likelihood for the Mallows approximation over all train and test splits for all trained datasets in cache/letor will be printed to the console. These would compare with the outputs appearing in /plots/letor/soi-Llog-all.txt when the main example is executed. 

Reproducibility
===
Included in the repository is code that faithfully reproduces the Tables and Figures in both the main paper and the supplement. In order to reproduce the Cayley Graph in Figure 1, simply run:

```python
python src/cayley.py
```

The code to reproduce the simulations performed in the paper and their corresponding figures is contained within the files ``src/laplacian_eigs.py`` and ``src/cdm_laplacian_eigs.py``. Specifically, the ``generate_PL_error_plot`` function in ``src/laplacian_eigs.py`` reproduces Figure 3, panel (a). Panels (b) and (c) are respectively reproduced by the functions ``generate_CRS_on_PL_err_plot`` and ``generate_CRS_err_plot`` in ``src/cdm_laplacian_eigs.py``. Finally, Figure 4 can be reproduced using the function ``generate_CRS_err_plot_variousn`` in ``src/cdm_laplacian_eigs.py``.

Instructions to reproduce the experiments on real data, notably Table 1 and Figure 2 are highlighted in the section above on fitting models to real data.

License
==
Released under the MIT license
