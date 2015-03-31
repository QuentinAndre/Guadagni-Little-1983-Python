# Recreation of the famous marketing paper by Guadagni and Little (1983) in Python
This git includes:
  1. **Dataset Recreation.py**: A script to create from scratch a purchase history dataset of N consumers over T time periods for K options. All the parameters in the script can be changed to generate a different dataset. As in the original paper, the consumers are heterogenous in their loyalty to the brand and to the different sizes offered. The script outputs four files:
    * *GuadagniLittle1983.csv*, which contains the data to estimate
    * *TrueBetas.csv*, which contains the true parameters used to generate the data.
    * *BrandShares.png*, which plots the evolution of market shares for the brands over time.
    * *SizeShares.png*, which plots the evolution of market shares for the sizes over time.
  2. **Mixed Logit Estimation.py**: A script to estimate the parameters used to generate the data. As in the original paper, the script constrains the utility of the first option to be 1, and estimates (K-1) brand intercepts and J utility components for the attributes, which are common to all brands.
