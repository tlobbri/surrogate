# Introduction

surrogate.py is a script in order to Interpolate/approximate a database using 
neural network based on pandas dataframe and scipy RBF function.
The script also computes the ANAOVA decomposition


# Usage

metaModels = Surrogate(Input, Output)
* 'Input'  is a dataframe of the input variables (eg. 'x_1', 'x_2', 'x_3')
* 'Output'  is a dataframe of the output functions (eg. 'f')

metaModels.train(kernel='multiquadric', smooth=0.8)

if smooth is equal to 0.0, the neural network is interpolating.


In order to evaluate the the meta-model the any parameter combination :

fmodel.append(metaModels.eval('f', {'x_1':x,'x_2':y,'x_3' : z}))

where, (x,y,z) are the numeric values of the respective parmeter

In order to compute the ANOVA decomposition

metaModels.computeANOVA()
metaModels.plotANOVA(saving=True)
print metaModels.getAnova()

# More?

see test() function for more detailed pieces of information
