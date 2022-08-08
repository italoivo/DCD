# DCD
Dynamic community detection and parameter search code

Description

This toolbox contains a python version of the Generalized Louvain Algorithm (Mucha et al 2010). Modified from the Louvain algorithm contained in the brain connectivity toolbox for python. In addition to the Generalized Louvain algorithm this toolbox brings an automated parameter detection algorithm based on skewness minimization and scale freeness optimization to determine the resolution parameters (Pinto et al, unpublished).

Usage

This toolbox aims to automatize dynamic community detection pipelines, starting from multi time layer connectivity matrices we provide a method to estimate the resolution parameters using the function parameter_search, once the resolution parameter are determined the user can obtain dynamic community structures by using the dynamic_louvain function.

Authors and acknowledgment

This project was developped and maintained by Italo Ivo Lima Dias Pinto with support of Javier Omar Garcia and Kanika Bansal.
