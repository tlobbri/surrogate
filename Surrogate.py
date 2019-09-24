# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:08:30 2019

@author: tlo
"""

import math
import pandas as pd
from scipy.interpolate import Rbf
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['grid.linewidth'] = 1.0
#mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] =  14
mpl.rcParams['ytick.labelsize'] =  14
mpl.rcParams['legend.fontsize'] = 16



class Surrogate():
    def __init__(self, dataInput, dataOutput):
        self.dataInput = dataInput
        self.dataOutput = dataOutput
        self.dataInput_scaled = None
        self.dataOutput_scaled = None
        
        self.surrogates = {}
        
        pass
    
    def _scaleInput(self):
        self.dataInputMin =self.dataInput.min()
        self.dataInputMax =self.dataInput.max()
        
        self.dataInput_scaled = self.dataInput.copy()
        
        for c in self.dataInput.columns:
            self.dataInput_scaled[c] = (self.dataInput_scaled[c]-self.dataInputMin[c])/(self.dataInputMax[c]-self.dataInputMin[c])
        
    def _scaleOutput(self):
        self.dataOutputMin =self.dataOutput.min()
        self.dataOutputMax =self.dataOutput.max()
        
        self.dataOutput_scaled = self.dataOutput.copy()
        
        for c in self.dataOutput.columns:
            self.dataOutput_scaled[c] = (self.dataOutput_scaled[c]-self.dataOutputMin[c])/(self.dataOutputMax[c]-self.dataOutputMin[c])
    
    
    def train(self, kernel='multiquadric', smooth=0.0):
        self._scaleInput()
        self._scaleOutput()

        nbExp = self.dataInput_scaled.values.shape[0]
        
        for OutputName in self.dataOutput.columns:
            _data = []
            print('Training ' + OutputName)
            #for iExp in range(nbExp):
            for iExp, irow in self.dataInput_scaled.iterrows():
                _ldata = []
                for InputName in self.dataInput.columns:
                    #print InputName, iExp
                    #print self.dataInput_scaled
                    _ldata.append(self.dataInput_scaled[InputName][iExp])
                    
                _ldata.append(self.dataOutput_scaled[OutputName][iExp])  
                _data.append(_ldata)            
                        
            
            _data= np.array(_data)
            #print _data
            eps= math.pow(1./nbExp,1./self.dataInput.columns.shape[0])
            #print eps
            self.surrogates[OutputName] = Rbf(*(_data).transpose(),function=kernel, epsilon=eps, smooth=smooth)
                

            
    
    def eval(self, outputName, X):
        Xscaled = []
        for c in self.dataInput.columns:
            Xscaled.append((X[c]-self.dataInputMin[c])/(self.dataInputMax[c]-self.dataInputMin[c]))
        return self.surrogates[outputName](*np.array(Xscaled))*(self.dataOutputMax[outputName]-self.dataOutputMin[outputName])+self.dataOutputMin[outputName]
    
    def _computeTotalMeanVar(self,outputName, N=1000):
        
        Y = np.zeros(N)
        for i in range(N):
            Xscaled = []
            for c in self.dataInput.columns:
                Xscaled.append(random.random())
            Y[i] = self.surrogates[outputName](*np.array(Xscaled))*(self.dataOutputMax[outputName]-self.dataOutputMin[outputName])+self.dataOutputMin[outputName]
            
        return [np.mean(Y), np.var(Y)]
    def _computeMeanFixedInput(self,outputName, fixedInputName, fixedInputValue, N=1000):
        
        Y = np.zeros(N)
        for i in range(N):
            Xscaled = []
            for c in self.dataInput.columns:
                if c == fixedInputName:
                    Xscaled.append(fixedInputValue)
                else:
                    Xscaled.append(random.random())
            Y[i] = self.surrogates[outputName](*np.array(Xscaled))*(self.dataOutputMax[outputName]-self.dataOutputMin[outputName])+self.dataOutputMin[outputName]
            
        return np.mean(Y)
    
    def getInput(self):
        return self.dataInput.copy()
    
    def getInputScaled(self):
        return self.dataInput_scaled.copy()

    def getOutput(self):
        return self.dataOutput.copy()
    
    def getOutputScaled(self):
        return self.dataOutput_scaled.copy()    

        
    def computeANOVA(self, outputList=None, N=500, NN=10000):
        print('computing ANOVA')
        self.anova = {}
        ##############################
        #iterate over each output
        ##############################
        io = 1
        for outputName in self.dataOutput.columns:
            self.anova[outputName]={}
            print('responses (%d/%d): ' % (io,self.dataOutput.columns.shape[0] ) + outputName)   
            io +=1
            #############################################
            #   Computing total mean and variance       #
            #############################################
            [totalMean, totalVar] = self._computeTotalMeanVar(outputName, N=NN)
            
            
            ######################################
            #    Iterate over each input         #
            ######################################
            
            sumAnova = 0
            ii = 1
            for inputName in self.dataInput.columns:
                print('input (%d/%d): '%(ii,self.dataInput.columns.shape[0]) + inputName)                
                ii+=1
                Y = np.zeros(N)
                for i in range(N):
                    inputValue = random.random()
                    Y[i] = (self._computeMeanFixedInput(outputName,inputName,inputValue)-totalMean)**2
                varInput = np.mean(Y)
                self.anova[outputName][inputName]= varInput/totalVar
                sumAnova += varInput/totalVar
            self.anova[outputName]['2^{nd}Order'] = 1-sumAnova
        return self.anova.copy()
                
                
    def getAnova(self):
        return self.anova.copy()
    
    def setAnova(self, anova):
        self.anova = anova.copy()
                
    def plotANOVA(self, saving=False, baseName=None, hist=True):
        
        for ko,vo in self.anova.iteritems():
            
            if hist:
                fig, ax = plt.subplots()
                import operator
                sorted_x = reversed(sorted(vo.items(), key=operator.itemgetter(1)))
    
                labels = []
                data = []
                
                for s in sorted_x:
                    labels.append(r'$%s$' % s[0])
                    data.append(s[1]*100)
                     
                y_pos = np.arange(len(labels))
                                
                ax.barh(y_pos, data, align='center', color='green', ecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()  # labels read top-to-bottom
                ax.set_xlabel(r'ANOVA $[\%$]')
                ax.set_title(r'$%s$' % ko)
                plt.gca().xaxis.grid(True)
                ax.set_xlim([0,100])
                
            else:

                labels = []
                for l in vo.keys():
                    labels.append(r'$%s$' % l)
                #print labels
                plt.figure()    
                plt.pie(vo.values(), labels=labels,startangle=140, autopct='%1.1f%%')
    
                plt.title(r'$%s$' % ko)
                plt.axis('equal')
                plt.tight_layout()
            if saving :
                if (baseName == None):
                    plt.savefig(ko + '.png')
                else:
                    plt.savefig(baseName +'\\' +  ko + '.png')
    

    def plotANOVA2(self, saving=False, baseName=None):
        
        for ko,vo in self.anova.iteritems():
            fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
            
            labels = []
            for l in vo.keys():
                labels.append(r'$%s$' % l)
            print labels
            data = vo.values()
            
            wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
            
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
                      bbox=bbox_props, zorder=0, va="center")
            
            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                             horizontalalignment=horizontalalignment, **kw)
            
            ax.set_title(ko)
            if saving :
                plt.savefig(baseName +'\\' + ko + '.png')

def test():
        
    def myfunction(x, y, z, err=0.0):
        return (x+y)*math.exp(2*z) + random.random()*err
    df = pd.DataFrame()
    
    nbExp = 100
    
    for iExp in range(nbExp):
        x = random.random()
        y = random.random()
        z = random.random()
        
        f =myfunction(x,y,z, 2)
        df = df.append({'x_1' : x,
                        'x_2' : y,
                        'x_3' : z,
                        'f' : f},ignore_index=True)
                        
       
    
    
    listInput = ['x_1','x_2','x_3']
    listOutput = ['f']
    
    
    def isIn(a,b):
        r = []
        for i in range(a.shape[0]):
            if a[i] in b:
                r.append(True)
            else:
                r.append(False)
        return np.array(r)
    
    Input = df.loc[:, isIn(df.columns,listInput)]
    Output = df.loc[:, isIn(df.columns,listOutput)]
    
    
    
    metaModels = Surrogate(Input, Output)
    metaModels.train(kernel='multiquadric', smooth=0.8)
    
    
    
    nbCheck = 10000
    
    xcheck =[]
    freal = []
    fmodel = []
    for iExp in range(nbCheck):
        x = random.random()
        y = random.random()
        z = random.random()
        
        freal.append(myfunction(x,y,z))
        fmodel.append(metaModels.eval('f', {'x_1':x,'x_2':y,'x_3' : z}))
        xcheck.append(x)
        
    freal= np.array(freal)
    fmodel= np.array(fmodel)
    
    
    """
    metaModels.computeANOVA()
    metaModels.plotANOVA(saving=True)
    print metaModels.getAnova()
    """

    
    print np.sum((freal-fmodel)**2/float(freal.shape[0]+1))
    
    plt.figure()
    plt.plot(xcheck, freal , 'o')
    plt.plot(xcheck, fmodel, '*')
    
if __name__=="__main__":
    test()