"""
Implementation of a Conditional/Multinomial Logit Estimation:

In this specification, we assume that all individuals are choosing among 
J different options possessing K characteristics. 

Each option j has a fixed utility component (Multinomial element), and each 
feature k has a fixed parameter common to all options (Conditional element).
The model therefore estimates J + K parameters.

This estimation model is battery-included:
the data and the true parameters used to generate the data are available on
GitHub: https://github.com/QuentinAndre/MarketingModel
@author: Quentin Andre
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
                   
class ChoiceModel(object):
    def __init__(self, data, Y, X, J):
        """
        We initialize the choice model from a Pandas Dataframe in which the
        data is presented in long form (one line per choice and per individual)
            -Data is the Pandas dataframe
            -Y is the name of the column containing the dummy exogenous variable,
            i.e. the choice
            -X is the list of columns containing the K endogenous variables.
            -J is the column identifying the choice option among J
        The model automatically generate J dummy variables representing 
        the fixed utility components for each option. The utility parameter of
        the first option is set up to be 0 to allow estimation.
        """
        #Generating the fixed utility columns
        i = 0
        for option in set(data[J].values):
            if i == 0:
                data["Constant"] = 1*(data[J] == option)
                X.insert(i, "Constant")
            else:
                data["Option_{}".format(option)] = 1*(data[J] == option)
                X.insert(i, "Option_{}".format(option))
            i += 1
            
        #All variables are stored
        self.varnames = X   
        
        #Storing the endogenous and exogenous components in the model
        self.endog = data.loc[:, Y].values
        self.exog = data.loc[:, X].values
        
        #Number of options
        self._selectors = set(data.loc[:, J])
        self.J = len(self._selectors)
        
        #Number of parameters
        self.K = self.exog.shape[1]
        
        #Number of individuals
        self.N = self.exog.shape[0]/self.J   
        
        #Generating an JxNxK array: this makes the optimization procedure faster,
        #as loops are not necessary anymore:
            #The first dimension is the option
            #The second dimension is the individual
            #The third dimension is the attribute.
        self._attriblists = []
        for s in self._selectors:
            self._attriblists.append(data.loc[data["Option"]==s,X].values)
        self.attribcubes = np.array(self._attriblists)
                        
    def cdf(self, XB):
        """
        CDF of the multinomial logit given a vector XB.
        Return the probability of choice of the option j for the N individuals
        in a JxN matrix, flattened into a (JxN)x1 array.
        """
        eXB = np.exp(XB)
        return (eXB/eXB.sum(0)).flatten(order='F')

    def loglike(self, params):
        """
        Log-likelihood of the choice model.
        The choice probabilities return by the cdf function are 
        log-transformed, and summed accross all individuals.
        Return the sum of the negative log-likelihoods of the individuals
        """
        params = params.reshape(self.K, -1, order='F')
        logprob = np.log(self.cdf(np.dot(self.attribcubes, params)))
        return -np.sum(self.endog * logprob)

    def loglikeobs(self, params):
        """
        Log-likelihood of the observations in the choice model.
        Returns the negative log-likelihood function of the JxN observations 
        evaluated at `params' (an observation is an Option x Individual pair).
        """
        params = params.reshape(self.K, -1, order='F')
        logprob = np.log(self.cdf(np.dot(self.attribcubes, params)))
        return (self.endog * logprob)

    def score(self, params):
        """
        Used to speed up the optimization process of the log-likelihood.
        Returns the score matrix.
        """
        params = params.reshape(self.K, -1, order='F')
        L = self.cdf(np.dot(self.attribcubes, params))
        return -np.dot(self.endog-L, self.exog)
        
    def hessian(self, params):
        """
        Return the Hessian of the likelihood model, evaluated at `params'.
        """
        params = params.reshape(self.K, -1, order='F')
        L = self.cdf(np.dot(self.attribcubes, params))
        return np.dot(L*(1-L)*self.exog.T,self.exog)

    def fit_likelihood(self, x0=None, approx_grad=False):
        """
        Fit the parameters to the data
        """
        #Constraining the parameter of the first option to be 0:
        constraints = [(0, 0)] + [(None, None)]*(self.K-1)
        
        #Fitting the parameters to the data
        if x0 == None:
            x0 = [0]*self.K
        if approx_grad==False:
            self.fitted = opt.fmin_l_bfgs_b(self.loglike, x0=x0, fprime=self.score, 
                                            bounds=constraints)
        else:
            self.fitted = opt.fmin_l_bfgs_b(self.loglike, x0=x0, approx_grad=True, 
                                            bounds=constraints)
        
        #Return the success (or failure) of the log-likelihood estimation
        if self.fitted[2]["warnflag"] != 0:
            return "The model could not converge!"
        else:
            return "Convergence achieved! Negative log-likelihood = {0}".format(self.fitted[1])
    
    def print_estimates(self, true_betas=False):
        """
        Obtain the significance level of the different parameters compared to 0. 
        If the option "true_betas" is specified, then the coefficients will be 
        compared to the true betas instead.
        Returns a Dataframe containing the results of the estimation
        """
        if self.fitted == None:
            return "The parameters must first be estimated!"
        #Storing the estimates in a Pandas dataframe
        norm_dist = stats.norm(loc=0, scale=1)     
        betas = self.fitted[0]
            
        #Obtention of the standard errors of the parameters estimates:     
        #Sandwich formula
        var_matrix = np.linalg.inv(self.hessian(betas))
        #The variance of the parameters are on the diagonal
        se_betas = np.diag(var_matrix)
        
        #Estimates are stored in a dataframe:
        self.results = pd.DataFrame(np.array([self.varnames, betas, se_betas]).T, 
                                    columns=["Coeff. Name", "Coeff. Est.", "Std. Err."])
                                    
        self.results = self.results.convert_objects(convert_numeric=True)
        
        #Hypothesis-testing, against the true betas or against zero:
        if true_betas is not False:
            self.results.insert(1, "True Coeff.", true_betas)
            self.results["t-stat"] = self.results.apply(lambda x: \
                            (x["Coeff. Est."]-x["True Coeff."])/x["Std. Err."], 
                                                        axis=1
                                                        )
        else:
            self.results["t-stat"] = self.results.apply(lambda x: \
                            (x["Coeff. Est."]-0)/x["Std. Err."], 
                                                        axis=1
                                                        )
        #Compute the p-values and 95% confidence intervals for the parameters
        self.results["p-value"] = self.results["t-stat"].apply(lambda x: \
                            norm_dist.cdf(-np.abs(x))*2
                                                              )
        self.results["95% Conf. Int"] = self.results.apply(lambda x: \
                                "[{0:.3f}  -  {1:.3f}]".format(
                                        x["Coeff. Est."] - x["Std. Err."]*1.96, 
                                        x["Coeff. Est."] + x["Std. Err."]*1.96
                                                              ), axis=1
                                                          )
                                                        
        #Return the DataFrame containing the results                                                
        return self.results
            


 
if __name__ == "__main__":
    # Data and true parameters, to be forked from GitHub
    data = pd.read_csv("GuadagniLittle1983.csv")
    true_betas = pd.read_csv("TrueBetas.csv", header=None)
    print(true_betas)
    guad_litt = ChoiceModel(data, "Chosen", ["BasePrice", "Promo", "PromoSize", "PrevPromo1",
                                 "PrevPromo2", "BLoyal", "SLoyal"], "Option")
    guad_litt.fit_likelihood()
    print("Estimates, compared to 0")
    print(guad_litt.print_estimates())
    print("Estimates, compared to true parameters")
    print(guad_litt.print_estimates(true_betas=true_betas))