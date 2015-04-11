
.. highlight:: sh

==============
 Introduction
==============
:Date: March 15, 2015
:Version: 1.0.0
:Authors: Quentin ANDRE, quentin.andre@insead.edu
:Web site: https://github.com/QuentinAndre/Guadagni-Little-1983-Python
:Copyright: This document has been placed in the public domain.
:License: Guadagni-Little-1983-Python is released under the MIT license.

Purpose
=======
Guadagni-Little-1983-Python provides a Python implementation of the mixed/conditional logit model of product choice
outlined by Guadagni and Little in their 1983 article "A Logit Model of Brand Choice Calibrated on Scanner Data".

Content
=======
1. **Dataset Recreation.py**: A script to create from scratch a purchase history dataset of N consumers over T time periods for K options. All the parameters in the script can be changed to generate a different dataset. As in the original paper, the consumers are heterogenous in their loyalty to the brand and to the different sizes offered. The script outputs four files:
* *GuadagniLittle1983.csv*, which contains the simulated scanner data
* *TrueBetas.csv*, which contains the true parameters used to generate the data.
* *BrandShares.png*, which plots the evolution of market shares for the brands over time.
* *SizeShares.png*, which plots the evolution of market shares for the sizes over time.

2. **Mixed Logit Estimation.py**: A script to recover the parameters used to generate the data. As in the original paper, the  the utility of the first option is constrained to be 1 to allow identification of the (K-1) brand intercepts and of the J utility components for the attributes (which are common to all brands).

Installation
============

Dependencies
------------
This code has been tested in Python 3.4, using the Anaconda distribution:
* `The Anaconda distribution for Python 3.4 <http://continuum.io/downloads#py34>`_

Download
--------

* Using git::

   git clone https://github.com/QuentinAndre/Guadagni-Little-1983-Python.git

* Download the master branch as a `zip archive
    <https://github.com/QuentinAndre/Guadagni-Little-1983-Python/archive/master.zip>`_


References
==========
Guadagni, P. M., & Little, J. D. (1983). A logit model of brand choice calibrated on scanner data. 
Marketing science, 2(3), 203-238.
