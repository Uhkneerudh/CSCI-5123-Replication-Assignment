# CSCI 5123: Recommender Systems Replication Assignment
For my final project for CSCI 5123, *Recommender Systems*, I replicated and extended a recent paper in the field of recipe recommendation. The paper, “HUMMUS: A
Linked, Healthiness-Aware, User-centered and ArgumentEnabling Recipe Data Set for Recommendation" poses a new
dataset with consolidated recipes, reviews, and features, that
can be used to recommend nutritious recipes. I replicated the dataset and experiments, recreating graphs, tables, and key metrics to confirm the original
paper’s results. I also extended the original work by
adding two new metrics, tuning model hyperparameters, and
introducing 5 new models.


dataset_scripts.ipynb: The core file used in replication and extension. The code to generate all figures and tables is in this file, and new code I wrote is clearly documented via Markdown Headers.

implicit_util.py: The file I modified to implement Hit Ratio and AUC. The code to do so is highlighted using comments.

irec_util.py: The file I modified to implement Hit Ratio and AUC. The code to do so is highlighted using comments.

Replication_Assignment.pdf: the PDF writeup of my research and findings, written in ACM format.