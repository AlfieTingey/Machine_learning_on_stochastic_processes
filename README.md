# Machine_Learning_Algorithms_Individual_Project

This is a repository of the different algorithms that I am using in my individual Project. 

# EM-Algorithm: 

Contains two folders, one based on the full (dense) MAP EM-algorithm implementation, and one with the ER-CHMM algorithm implementation. To use, navigate into the folder with LINE toolbox and put into the examples section. This is required for all scripts. The real data traces are also present in this folder.

The evaluation scripts evaluate the algorithms using input trace data that a user can define. 

# Gibbs Sampling: 

We have function that implements the Gibbs sampling algorithm for demand estimation and another script which evaluates the Gibbs sampling using set examples. Users can create any examples that they want in there. 

# RNN Methodology: 

Contains python scripts that implement the RNN methodology, scripts that can be used to generate traces from queueing networks, and another script that evaluates the predicted topology of the queueing networks. The RNN algorithm will save the learned parameters in a .txt file in the 'models learned' folder. Within the 'main.py' script a user has to specify the directory thst they want to save their files, and specify the directory that contains the learned traces. We have included some traces that we generated from queueing networks that a user can use. 

# Sources and References: 

Gibbs Sampling:
Casale, G. Wang, W. & Sutton, C. (2016). A Bayesian Approach to Parameter Inference in Queueing Networks. ACM Trans. Model. Comput. Simul.,27(1).

Full MAP EM-Algorithm: 

Buchholz, P. An EM Algorithm for MAP Fitting from Real Traffic Data. (2003). Springer-Verlag. Berlin.

ER-CHMM EM-Algorithm:

Okamura, H. et al. (2008). An EM algorithm for a Superposition of Markovian Arrival Processes. Department of Information Engineering, Graduate School of Engineering,
Hiroshima University, Japan. Accessible at: http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1589-28.pdf.

Horvath, G. & Okamura, H. (2013). A Fast EM algorithm for Fitting Marked Markov Arrival Processes with a new Special Structure.

Horvath, G. et al. (2018). Parallel Algorithms for Fitting Markov Arrival Processes. Preprint Submitted to Elsevier.

This algorithm was inspired and adapted from from the butools implementation, accessible at: 
http://webspn.hit.bme.hu/~telek/tools/butools/doc/MAPFromTrace.html

RNN methodology: 

This code has been inspired and adapted from the following source:
Garbi, G et al. (2020). Learning Queueing Networks by Recurrent Neural Networks.
Accessible at:
https://pdfs.semanticscholar.org/7f7c/12bcc23ba098ad5a4a0ad251bd92e9b9c27a.pdf.







