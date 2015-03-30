"""
Recreation of a dataset similar to Guadagni and Little (1983), in which
the price and the promotion status of several options of different brands
and sizes vary over time.

The individual are choosing among brand and sizes, and are heterogenous in
their loyalty for a particular brand and a particular size.

The parameters can be tweaked to generate a different dataset. The estimation
procedure developed is available on GitHub: 

https://github.com/QuentinAndre/MarketingModel

@author: Quentin Andre
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools
from math import exp
import seaborn as sns
import matplotlib.pyplot as plt
import csv

"""
Parameters: the following can be changed to create a different dataset !
"""
# Number of households
N = 600
#Number of time_periods:
T = 120

# Number of sizes and brands available
V_Sizes = np.array([0, 1])
V_Brands = np.array([0, 1, 2, 3, 4])

# Here we define several important constants used to generate the data.
# 
# * $\alpha_B: \text{Smoothing constant of the loyalty for brand}$
# * $\alpha_S: \text{Smoothing constant of the loyalty for size}$
# 
# * $M_{PriceM}: \text{Price mean matrix for the Brand/Size pairs}$
# * $M_{PriceSD}: \text{Price SD matrix for the Brand/Size pairs}$
# 
# * $M_{PromoP}: \text{Probability that a Brand/Size pair is promoted on a given week}$
# * $M_{PromoM}: \text{Promotion discount mean matrix for the Brand/Size pairs}$
# * $M_{PromoSD}: \text{Promotion discount SD matrix for the Brand/Size pairs}$

alpha_B = 0.875
alpha_S = 0.812

#Price is expressed in $/oz in the paper
M_priceM = np.array(
                    [[3.25, 2.75, 2.65, 4.20, 1.95],
                     [3.00, 2.50, 2.1, 1.5, 2.1]]
                    )
                    
#We assume very small price variations
M_priceSD = np.array(
                      [[0.12, 0.05, 0.06, 0.07, 0.09],
                      [0.03, 0.05, 0.07, 0.09, 0.01]]
                     ) 
#Probability of observing a promotion
M_promoP = np.array(
                    [[0.25, 0.35, 0.15, 0.20, 0.12], 
                     [0.14, 0.15, 0.18, 0.21, 0.35]]
                    )

#Promotion is expressed in percentage
M_promoM = np.array(
                    [[0.15, 0.19, 0.12, 0.26, 0.32],
                     [0.17, 0.28, 0.07, 0.12, 0.4]]
                    ) 
 #We assume very small variations in magnitude
M_promoSD = np.array(
                    [[0.03, 0.01, 0.02, 0.06, 0.01], 
                     [0.01, 0.03, 0.05, 0.05, 0.04]]
                     )


# Here we define the utility function and utility parameters of the household 
# i for the brand-size option k at times t:

# $U_{i, k, t}= \beta^{0}x^{0}_{i, k} + \beta^{1}x^{1}_{k, t} + 
# \beta^{2}x^{2}_{k, t} + \beta^{3}x^{3}_{k, t} + \beta^{4}x^{4}_{i, k, t} + 
# \beta^{5}x^{5}_{i, k, t} + \beta^{6}x^{6}_{i, k, t} + \beta^{7}x^{7}_{i, k, t} 
# +\epsilon_{i, t}$
# 
# * $\beta^{0}$: Fixed utility constants of the brand-size pairs
# * $\beta^{1}$: Price factor
# * $\beta^{2}$: Promotion factor
# * $\beta^{3}$: Promotion discount factor
# * $\beta^{4}$: Previous promotion purchase factor
# * $\beta^{5}$: Second previous promotion purchase factor
# * $\beta^{6}$: Brand Loyalty factor
# * $\beta^{7}$: Size Loyalty factor


# Made-up coefficients:
# They ensure that the multinomial logit estimation can recover estimates 
# (enough variability in choices)

b0 = np.array([[0, 0.3, 0.5, 0.6, 0.4],[0.7, 0.1, 0.3, 0.1, 0.2]])
b1 = -1.7
b2 = 0.3
b3 = 1.2
b4 = -0.5
b5 = -0.2
b6 = 2.5
b7 = 1.6

export = np.hstack((b0.flatten(), [b1, b2, b3, b4, b5, b6, b7]))
np.savetxt("Data Files\\TrueBetas.csv", export.T, delimiter=",")

"""
Initialization of the dataframes
"""

# Here we initialize four dataframes:
# 
# * $DF_{Supply}: \text{Panel data of Brand/Size pairs characteristics across weeks}$
# * $DF_{Demand}: \text{Panel data of household demand across weeks}$
# * $DF_{BLoyalty}: \text{Panel data of the Brand loyalties across households and weeks}$
# * $DF_{SLoyalty}: \text{Panel data of the Brand loyalties across households and weeks}$
# 

# Supply Dataframe
weeks = [i for i in range (1, T+1)]
week_pairs_list = [i for i in itertools.product(weeks, V_Sizes, V_Brands)]

weeks_list = pd.DataFrame([i[0] for i in week_pairs_list])
sizes_list = pd.DataFrame([i[1] for i in week_pairs_list])
brands_list = pd.DataFrame([i[2] for i in week_pairs_list])
price_list = promo_list = promosize_list = pd.DataFrame([np.NaN])

DF_Supply = pd.concat([weeks_list, sizes_list, brands_list, price_list, 
                       promo_list, promosize_list], axis=1)

colsTime = ["Week", "Size", "Brand", "BasePrice", "Promo", "PromoSize"]
DF_Supply.columns = colsTime

def gen_random_price(x):
    """
    Given the Size and the Brand, return a
    random price for the time period
    """
    sizeid = x["Size"]
    brandid = x["Brand"]
    price_mean = M_priceM[sizeid][brandid]
    price_SD = M_priceSD[sizeid][brandid]
    return stats.norm.rvs(price_mean, price_SD)

def gen_random_promo(x):
    """
    Given the Size and the Brand, return a
    discount or not for the time period
    """
    sizeid = x["Size"]
    brandid = x["Brand"]
    promo_prob = M_promoP[sizeid][brandid]
    return stats.bernoulli.rvs(promo_prob)

def gen_random_discount(x):
    """
    If the product is discounted on that time period,
    return a discount amount given the Size and the Brand
    """
    if x["Promo"] == 0:
        return 0
    else:
        sizeid = x["Size"]
        brandid = x["Brand"]
        promo_mean = M_promoM[sizeid][brandid]
        promo_SD = M_promoSD[sizeid][brandid]
        return stats.norm.rvs(promo_mean, promo_SD)


DF_Supply["Option"] = DF_Supply["Size"]*5 + DF_Supply["Brand"]
DF_Supply["BasePrice"] = DF_Supply.apply(gen_random_price, axis=1)
DF_Supply["Promo"] = DF_Supply.apply(gen_random_promo, axis=1)
DF_Supply["PromoSize"] = DF_Supply.apply(gen_random_discount, axis=1)


# Demand Dataframe
households = [i for i in range(1, N+1)]
households_weeks_list = [i for i in itertools.product(weeks, households)]

weeks_list = pd.DataFrame([i[0] for i in households_weeks_list])
households_list = pd.DataFrame([i[1] for i in households_weeks_list])
sizechoice_list = brandchoice_list = pd.DataFrame([0])

DF_Demand = pd.concat([weeks_list, households_list, sizechoice_list, brandchoice_list], axis=1)
colsPurchase = ["Week", "Household","Size", "Brand"]
DF_Demand.columns = colsPurchase

DF_Demand.loc[DF_Demand.Week==1, "Size"] = np.random.choice([0, 1], size=N)
DF_Demand.loc[DF_Demand.Week==1, "Brand"] = np.random.choice([0, 1, 2, 3, 4], size=N)
DF_Demand.loc[DF_Demand.Week==2, "Size"] = np.random.choice([0, 1], size=N)
DF_Demand.loc[DF_Demand.Week==2, "Brand"] = np.random.choice([0, 1, 2, 3, 4], size=N)
DF_Demand.loc[:, "Option"] =  DF_Demand["Size"]*5 + DF_Demand["Brand"]

def update_prev_promo(week, past=1):
    """
    Return the brand code if the past purchase was a promotional purchase, otherwise returns a number greater than 10
    """
    colname = "PrevPromo{0}".format(past)
    promo_history = pd.DataFrame(DF_Supply.loc[DF_Supply.Week==(week-past), 
                                               ["Brand", "Promo"]]).groupby(["Brand"]).sum().reset_index()
    if week == 3:
        DF_Demand[colname] = np.NaN #Initialize the past brand column
    
    out = pd.merge(DF_Demand.loc[DF_Demand.Week==(week-past)], promo_history, on=["Brand"], how="left").sort(["Week", "Household"])
    DF_Demand.loc[DF_Demand.Week==week, colname] = np.array(out["Promo"]==0)*10 + np.array(out["Brand"])
    
    
def update_all(week):
    """
    Update all the previous period variables, as well as the loyalty
    """
    update_prev_promo(week, past=1)
    update_prev_promo(week, past=2)
    update_BLoyalty(week)
    update_SLoyalty(week)
    
   
def gen_choices(week):
    """
    Generate the utility for brand-size options for a given week, and generate a choice of size and brand for all households.
    """
    #Updating the Demand DataFrame to reflect past choices
    update_all(week)
    mask_week = (DF_Supply["Week"] == week)
    
    #Dataframe of utilities, one row per household
    utilities = pd.DataFrame([[0]*10]*N, columns=["Option_{0}".format(i) for i in range(0, 10)])
    current_week = DF_Demand.loc[DF_Demand.Week==week]
    
    #Looping over sizes and brands to create the utility  for each option
    for size in V_Sizes:
        mask_size = (DF_Supply["Size"] == size)
        for brand in V_Brands:
            option_number = size*5+brand
            mask_brand = (DF_Supply["Brand"] == brand)
            
            #Product infos from the Supply dataframe
            product_infos = DF_Supply[mask_week & mask_brand & mask_size]
            
            #Option-specific utility components (Scalar)
            utility1 = b0[size][brand] + b1*product_infos.iloc[0]["BasePrice"] + \
            b2*product_infos.iloc[0]["Promo"] + b3*product_infos.iloc[0]["PromoSize"]
            
            #Household-option-specific utility components (Vector of households)
            utility2 = b4*(current_week["PrevPromo1"]==brand) + b5*(current_week["PrevPromo2"]==brand) + \
            b6*DF_BLoyalty.loc[DF_BLoyalty.Week==week, brand] + \
            b7*DF_SLoyalty.loc[DF_SLoyalty.Week==week, size]
            
            #Assigning the utility in the matrix
            utilities["Option_{0}".format(option_number)] = utility1 + np.array(utility2)
    
    
    #Converting the utilities to exp-utilities
    utilities = utilities.applymap(exp)
    utilities["TOTAL"] = utilities.apply(sum, axis=1)
    
    #Converting the utilities to choice probabilities   
    proba = utilities.apply(lambda x: x/x["TOTAL"], axis=1).iloc[:,:-1]
    
    #Formulating the choice: random sampling of options weighted by choice probabilities
    choice = proba.apply(lambda x: np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                    p=np.array(x)), axis=1)
    
    #Updating the dataframe with the choices
    DF_Demand.loc[DF_Demand.Week==week, "Option"] = np.array(choice)
    DF_Demand.loc[DF_Demand.Week==week, "Size"] = np.array(choice)//5
    DF_Demand.loc[DF_Demand.Week==week, "Brand"] = np.array(choice)%5


# Size and Brand Loyalty Dataframes
households = [i for i in range(1, N+1)]
households_weeks_list = [i for i in itertools.product(weeks, households)]

weeks_list = pd.DataFrame([i[0] for i in households_weeks_list])
households_list = pd.DataFrame([i[1] for i in households_weeks_list])

DF_BLoyalty = pd.concat([weeks_list, households_list], axis=1)
DF_BLoyalty.columns=["Week", "Household"]
for brand in V_Brands:
    DF_BLoyalty[brand] = np.NaN
DF_BLoyalty.columns=["Week", "Household"]+list(V_Brands)
    
DF_SLoyalty = pd.concat([weeks_list, households_list], axis=1) 
DF_SLoyalty.columns=["Week", "Household"]
for size in V_Sizes:
    DF_SLoyalty[size] = np.NaN
DF_SLoyalty.columns=["Week", "Household"]+list(V_Sizes)


for brand in V_Brands:
    DF_BLoyalty.loc[DF_BLoyalty.Week==1, brand] = \
    alpha_B*(DF_Demand.loc[DF_Demand.Week==1, "Brand"]==brand) + \
    (1-alpha_B)/(V_Brands.shape[0]-1)*(DF_Demand.loc[DF_Demand.Week==1, "Brand"]!=brand)

for size in V_Sizes:
    DF_SLoyalty.loc[DF_SLoyalty.Week==1, size] = \
    alpha_S*(DF_Demand.loc[DF_Demand.Week==1, "Size"]==size) + \
    (1-alpha_S)*(DF_Demand.loc[DF_Demand.Week==1, "Size"]!=size)


def update_BLoyalty(week=2):
    for brand in V_Brands:
        new_loyalty = np.array(alpha_B*DF_BLoyalty.loc[DF_BLoyalty.Week==(week-1), brand] + \
        (1-alpha_B)*(DF_Demand.loc[DF_Demand.Week==(week-1), "Brand"]==brand))
        DF_BLoyalty.loc[DF_BLoyalty.Week==week, brand] = new_loyalty

def update_SLoyalty(week=2):
    for size in V_Sizes:
        new_loyalty = np.array(alpha_S*DF_SLoyalty.loc[DF_SLoyalty.Week==(week-1), size] + \
        (1-alpha_S)*(DF_Demand.loc[DF_Demand.Week==(week-1), "Size"]==size))
        DF_SLoyalty.loc[DF_SLoyalty.Week==week, size] = new_loyalty
        

update_BLoyalty(week=2)
update_SLoyalty(week=2)


for week in range(3, T+1):
    gen_choices(week)


# Putting this together: the purchase history
weeks = [i for i in range(3, T+1)]
households = [i for i in range(1, N+1)]
options = [i for i in range(0, 10)]
cols_id = [i for i in itertools.product(weeks, households, options)]

weeks_cols = pd.DataFrame([i[0] for i in cols_id])
households_cols = pd.DataFrame([i[1] for i in cols_id])
options_cols = pd.DataFrame([i[2] for i in cols_id])

DF_PurchaseHistory = pd.concat([weeks_cols, households_cols, options_cols], axis=1)
DF_PurchaseHistory.columns = ["Week", "Household", "Option"]


DF_PurchaseHistory = pd.merge(DF_PurchaseHistory, DF_Supply, on=["Week", "Option"], how="left").loc[:, ["Week", "Household",
                                                                                               "Size", "Brand", "Option", "BasePrice", 
                                                                                                "Promo", "PromoSize"]]
                                                                                                
DF_PurchaseHistory = pd.merge(DF_PurchaseHistory, DF_Demand, 
                              on=["Household", "Week", "Size", "Brand", "Option"], 
                              how="left").loc[:,["Week", "Household", "Size", "Brand", "Option", "BasePrice", "Promo", "PromoSize"
                                                 ,"PrevPromo1"]]
DF_PurchaseHistory["Chosen"] = ~np.isnan(DF_PurchaseHistory["PrevPromo1"])


DF_SLoyalty.columns = ["Week", "Household"] + ["SLoyal{0}".format(i) for i in V_Sizes]
DF_BLoyalty.columns = ["Week", "Household"] + ["BLoyal{0}".format(i) for i in V_Brands]
DF_PurchaseHistory = pd.merge(DF_PurchaseHistory, DF_SLoyalty, on=["Week", "Household"], how="left")
DF_PurchaseHistory = pd.merge(DF_PurchaseHistory, DF_BLoyalty, on=["Week", "Household"], how="left")

DF_PurchaseHistory["BLoyal"] = DF_PurchaseHistory.apply(lambda x: x["BLoyal{0}".format(int(x["Brand"]))], axis=1)
DF_PurchaseHistory["SLoyal"] = DF_PurchaseHistory.apply(lambda x: x["SLoyal{0}".format(int(x["Size"]))], axis=1)

DF_PurchaseHistory = DF_PurchaseHistory.loc[:, ["Week", "Household", "Option", "Chosen", "Size", "Brand", "BasePrice", "Promo", 
                                                "PromoSize", "SLoyal", "BLoyal"]]


prevpromo1 = DF_Supply.loc[:,["Week", "Brand", "Size", "Promo"]]
prevpromo1.columns = ["Week", "Brand", "Size", "LaggedPromo1"]
prevpromo1["Week"] = prevpromo1["Week"]+1
prevpromo1 = prevpromo1.groupby(["Week", "Brand"]).sum().reset_index()
prevpromo1 = prevpromo1.loc[:,["Week", "Brand", "LaggedPromo1"]].drop_duplicates(["Week", "Brand"])

prevpromo2 = DF_Supply.loc[:,["Week", "Brand", "Size", "Promo"]]
prevpromo2["Week"] = prevpromo2["Week"]+2
prevpromo2.columns = ["Week", "Brand", "Size", "LaggedPromo2"]
prevpromo2.groupby(["Week", "Brand"]).sum().reset_index()
prevpromo2 = prevpromo2.loc[:,["Week", "Brand", "LaggedPromo2"]].drop_duplicates(["Week", "Brand"])


prevchoice1 = DF_PurchaseHistory.loc[:,["Week", "Household", "Brand", "Size", "Chosen"]]
prevchoice1.columns = ["Week", "Household", "Brand", "Size", "LaggedChoice1"]
prevchoice1["Week"] = prevchoice1["Week"]+1
prevchoice1 = prevchoice1.groupby(["Week", "Household", "Brand"]).sum().reset_index()
prevchoice1 = prevchoice1.loc[:,["Week", "Household", "Brand", "LaggedChoice1"]].drop_duplicates(["Week", "Household", "Brand"])

prevchoice2 = DF_PurchaseHistory.loc[:,["Week", "Household", "Brand", "Size", "Chosen"]]
prevchoice2.columns = ["Week", "Household", "Brand", "Size", "LaggedChoice2"]
prevchoice2["Week"] = prevchoice2["Week"]+2
prevchoice2 = prevchoice2.groupby(["Week", "Household", "Brand"]).sum().reset_index()
prevchoice2 = prevchoice2.loc[:,["Week", "Household", "Brand", "LaggedChoice2"]].drop_duplicates(["Week","Household", "Brand"])


DF_PurchaseHistory = pd.merge(DF_PurchaseHistory, prevpromo1, on=["Week", "Brand"], how="left")
DF_PurchaseHistory = pd.merge(DF_PurchaseHistory, prevpromo2, on=["Week", "Brand"], how="left")
DF_PurchaseHistory = pd.merge(DF_PurchaseHistory, prevchoice1, on=["Week", "Household", "Brand"], how="left")
DF_PurchaseHistory = pd.merge(DF_PurchaseHistory, prevchoice2, on=["Week", "Household", "Brand"], how="left")


DF_PurchaseHistory["PrevPromo1"] = 1*(DF_PurchaseHistory["LaggedPromo1"]>0)&(DF_PurchaseHistory["LaggedChoice1"]>0)
DF_PurchaseHistory["PrevPromo2"] = 1*(DF_PurchaseHistory["LaggedPromo2"]>0)&(DF_PurchaseHistory["LaggedChoice2"]>0)


def bool_to_int(x):
    if x == False:
        return 0
    else:
        return 1

DF_PurchaseHistory["PrevPromo1"] = DF_PurchaseHistory["PrevPromo1"].apply(bool_to_int)
DF_PurchaseHistory["PrevPromo2"] = DF_PurchaseHistory["PrevPromo2"].apply(bool_to_int)

DF_PurchaseHistory = DF_PurchaseHistory.loc[DF_PurchaseHistory.Week>4, ["Week", "Household", "Option", "Chosen", "Size", "Brand", 
                                                                        "Option", "BasePrice", "Promo", "PromoSize", 
                                                                        "SLoyal", "BLoyal", "PrevPromo1", "PrevPromo2"]]

DF_PurchaseHistory["Chosen"] = 1*DF_PurchaseHistory["Chosen"]
DF_PurchaseHistory["ID"] = DF_PurchaseHistory["Week"] * 100 + DF_PurchaseHistory["Household"]


# Exportation to csv
DF_PurchaseHistory.loc[DF_PurchaseHistory.Week>10].to_csv("Data Files\\GuadagniLittle1983.csv", index=False)

# Plot of the size and brand market shares across weeks, saved to file for ref.
brand_share = DF_PurchaseHistory.loc[:,["Week", "Household", "Chosen", "Brand"]].iloc[:,0:4].groupby(["Week", "Household", "Brand"]).sum()
brand_share = brand_share.reset_index()
sns.tsplot(brand_share, time="Week", unit="Household", condition="Brand", value="Chosen", estimator=np.mean)
plt.savefig("Brand Shares.png")
plt.show()

size_share = DF_PurchaseHistory.loc[:,["Week", "Household", "Chosen", "Size"]].iloc[:,0:4].groupby(["Week", "Household", "Size"]).sum()
size_share = size_share.reset_index()
sns.tsplot(size_share, time="Week", unit="Household", condition="Size", value="Chosen", estimator=np.mean)
plt.savefig("Size Shares.png")
