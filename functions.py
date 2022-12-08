# Required function for running this assignment
# Written by Mehdi Rezvandehy


import numpy as np
import pandas as pd
import math
from matplotlib.offsetbox import AnchoredText
from typing import Callable
from scipy.stats import gaussian_kde
from IPython.display import display, Math, Latex
from matplotlib.ticker import PercentFormatter
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from IPython.display import HTML
import scipy
import itertools


class stats:
    def bootstrap_1D_replicate (val: list, func: Callable,seed: int=42) -> float:
        """ Generate bootstrap replicate of 1D data."""
        np.random.seed(seed)
        bs_sample=np.random.choice(val,len(val))
        return func(bs_sample)
        
    #########################  
    
    def bootstrap_2D_replicate (val1: list, val2: list,seed: int=42) -> list:
        """ Generate bootstrap replicate of 2D data."""
        np.random.seed(seed)
        inds = np.arange(len(val1))
        bs_inds = np.random.choice(inds, len(inds))
        bs_val1 = list(val1[bs_inds])
        bs_val2 = list(val2[bs_inds])   
        return bs_val1,bs_val2
        
    #########################  
    
    def LUSim(value: list,corr: [float],nsample:int, nsim:int,seed:int) -> [float]:
        """
        Simulate Gaussian correlated realization
        
        value   : distribution of each variable
        corr    : correlation matrix between variable
        nsample : number of sampling from data ditribution    
        nsim    : number of simulation
        
        """
        np.random.seed(seed)
        t_dist = scipy.stats.t(seed)
        matrix=corr
        L=scipy.linalg.cholesky(matrix, lower=True, overwrite_a=True)
        mu=0; sigma=1; nvar=len(matrix)
        w=np.zeros((nvar,nsample)) 
        N_Sim_R_val=[]
        for isim in range(nsim):
            # LU Simulation for Standard Gaussian
            for i in range (nvar):
                for j in range(nsample):
                    Dist = np.random.normal(mu, sigma, nsample)
                    w[i,:]=Dist
            Sim_R_val=[]
            N_var=[]
            for i in range(nsample):
                tmp=(np.matmul(L,w[:,i]))
                N_var.append(tmp)       
            N_var=np.array(N_var).transpose()   
            Sim_R_val=[]
            for i1 in range(nvar):
                R_tmp=[]
                for i2 in range(nsample):
                    prob=t_dist.cdf(N_var[i1][i2])
                    R_tmp.append(np.quantile(value[:,i1], prob, axis=0, keepdims=True)[0])
                Sim_R_val.append(R_tmp)  
    
            N_Sim_R_val.append(Sim_R_val)
            
        return N_Sim_R_val    
        
    #########################  
    
    def sign_effect(data_1: list, data_2: list,x:int =22,y:int =0.03,sign_level:float=0.1,
            fontsize: float=9,txt1: list= 'Not significant difference between the means',
            txt2:list ='Significant difference between the means'):
        """
        t-statistic (two sided t-test) and effect size calculation between two groups
        
        Effect\,Size=\frac{\mu _{1}-\mu _{2}}{\sigma_{pooled}}$
    
        where $\mu _{1}$ is the mean of first group and $\mu _{1}$ is the mean of 
        second group and $\sigma_{pooled}$ is the standard deviation of the population 
        from which the groups were sampled. It is also called **Pooled Standard Deviation** 
        : $\sigma_{pooled}=\sqrt{\frac{\sigma_{1}^2+\sigma_{2}^2}{2}}$
        """
        
        t_stats=ttest_ind(data_1, data_2)
        u1=np.mean(data_1)
        u2=np.mean(data_2)
        s1=np.var(data_1)
        s2=np.var(data_2)
        s_pooled=np.sqrt((s1+s2)/2)
        eff_size=abs((u1-u2)/s_pooled)
        if (eff_size<0.2):
            sign='Trivial'
        elif(eff_size>=0.2 and eff_size<0.5):
            sign='Small'        
        elif(eff_size>=0.5 and eff_size<0.8):
            sign='Medium'        
        elif(eff_size>=0.8 ):
            sign='Large'           
        if(t_stats[1]>sign_level):
            #txt+='\n t-statistic= '+str(np.round(t_stats[0],2))+ ', p-value= '+\
            txt1+='\np-value= '+str(np.round(t_stats[1],3))+'>'+str(sign_level)
            plt.text(x,y, txt1, color='r',fontsize=fontsize,bbox=dict(facecolor='white', alpha=0.2))
        else:
            #txt+='\n (t-statistic= '+str(np.round(t_stats[0],2))+ ', p-value= '+\
            txt2+='\np-value= '+str(np.round(t_stats[1],3))+'<'+str(sign_level)
            txt2+='\nEffect Size= '+sign+' ('+str(np.round(eff_size,3))+')'
            plt.text(x,y, txt2, color='r',fontsize=fontsize,bbox=dict(facecolor='white', alpha=0.2))    


##########################################################################################################
class EDA_plot:
    def histplt (val: list,bins: int,title: str,xlabl: str,ylabl: str,xlimt: list,
                 ylimt: list=False, loc: int =1,legend: int=1,axt=None,days: int=False,
                 class_: int=False,scale: int=1,int_: int=0,nsplit: int=1,
                 font: int=5,color: str='b') -> None :
        
        """ Make histogram of data"""
        
        ax1 = axt or plt.axes()
        font = {'size'   : font }
        plt.rc('font', **font) 
        
        #val=val[~np.isnan(val)]
        val=np.array(val)
        plt.hist(val, bins=bins, weights=np.ones(len(val)) / len(val),ec='black',color=color)
        n=len(val[~np.isnan(val)])
        Mean=np.nanmean(val)
        Median=np.nanmedian(val)
        SD=np.sqrt(np.nanvar(val))
        Max=np.nanmax(val)
        Min=np.nanmin(val)
    
        if (int_==0):
            txt='n=%.0f\nMean=%0.2f\nMedian=%0.1f\nσ=%0.1f\nMax=%0.1f\nMin=%0.1f'
        else:
            txt='n=%.0f\nMean=%0.2f\nMedian=%0.1f\nMax=%0.0f\nMin=%0.0f'        
        anchored_text = AnchoredText(txt %(n,Mean,Median,SD,Max,Min), borderpad=0, 
                                     loc=loc,prop={ 'size': font['size']*scale})    
        if(legend==1): ax1.add_artist(anchored_text)
        if (scale): plt.title(title,fontsize=font['size']*(scale+0.15))
        else:       plt.title(title)
        plt.xlabel(xlabl,fontsize=font['size']) 
        ax1.set_ylabel('Frequency',fontsize=font['size'])
        if (scale): ax1.set_xlabel(xlabl,fontsize=font['size']*scale)
        else:       ax1.set_xlabel(xlabl)
    
        try:
            xlabl
        except NameError:
            pass    
        else:
            if (scale): plt.xlabel(xlabl,fontsize=font['size']*scale) 
            else:        plt.xlabel(xlabl)   
            
        try:
            ylabl
        except NameError:
            pass      
        else:
            if (scale): plt.ylabel(ylabl,fontsize=font['size']*scale)  
            else:         plt.ylabel(ylabl)  
            
        if (class_==True): plt.xticks([0,1])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        ax1.grid(linewidth='0.1')
        if days:plt.xticks(range(0,45,nsplit),y=0.01, fontsize=8.6)  
        plt.xticks(fontsize=font['size']*scale)    
        plt.yticks(fontsize=font['size']*scale)   
        try:
            xlimt
        except NameError:
            pass  
        else:
            plt.xlim(xlimt) 
            
        try:
            ylimt
        except NameError:
            pass  
        else:
            plt.ylim(ylimt)        
    
    ######################################################################### 
            
    def KDE(xs: list,data_var: list,nvar: int,clmn: [str],color: [str],xlabel: str='DE Length',
            title: str='Title',ylabel: str='Percentage',LAMBA: float =0.3,linewidth: float=2.5,
            loc: int=0,axt=None,xlim: list=(0,40),ylim: list=(0,0.1),x_ftze: float =13,
            y_ftze: float=13,tit_ftze: float=13,leg_ftze: float=9) -> None :
        
        """
        Kernel Density Estimation (Smooth Histogram)
         
        """
        ax1 = axt or plt.axes()
        var_m=[]
        var_med=[]
        var_s=[]
        var_n=[]
        s1=[]
        data_var_=np.array([[None]*nvar]*len(xs), dtype=float)
        # Loop over variables
        for i in range (nvar):
            data = data_var[i]
            var_m.append(np.mean(data).round(2))
            var_med.append(np.median(data).round(2))
            var_s.append(np.var(data).round(1))
            var_n.append(len(data))
            density = gaussian_kde(data)
            density.set_bandwidth(LAMBA)
            density_=density(xs)/sum(density(xs))
            data_var_[:,i]=density_
            linestyle='solid'
            plt.plot(xs,density_,color=color[i],linestyle=linestyle, linewidth=linewidth)
            
        #############
        
        data_var_tf=np.array([[False]*nvar]*len(data_var_))
        for j in range(len(data_var_)):
            data_tf_t=[]
            for i in range (nvar):
                if (data_var_[j,i]==max(data_var_[j,:])):
                    data_var_tf[j,i]=True     
        #############            
        for i in range (nvar):
            plt.fill_between(np.array(xs),np.array(data_var_[:,i]),where=np.array(data_var_tf[:,i]),
                             color=color[i],alpha=0.9,label=clmn[i]+': n='+str(var_n[i])+
                             ', mean= '+str(var_m[i])+', median= '+str(var_med[i])+
                             ', '+r"$\sigma^{2}$="+str(var_s[i]))
        
        plt.xlabel(xlabel,fontsize=x_ftze, labelpad=6)
        plt.ylabel(ylabel,fontsize=y_ftze)
        plt.title(title,fontsize=tit_ftze)
        plt.legend(loc=loc,fontsize=leg_ftze,markerscale=1.2)
        
        ax1.grid(linewidth='0.2')
        plt.xlim(xlim) 
        plt.ylim(ylim) 

    ######################################################################### 
                        
    def CDF_plot(data_var: list,nvar: int,label:str,colors:str,title:str,xlabel:str,
                 ylabel:str='Cumulative Probability', bins: int =1000,xlim: list=(0,100),
                 ylim: list=(0,0.01),linewidth: float =2.5,loc: int=0,axt=None,
                 x_ftze: float=12,y_ftze: float=12,tit_ftze: float=12,leg_ftze: float=9) -> None:
        
        """
        Cumulative Distribution Function
         
        """
        ax1 = axt or plt.axes() 
        def calc(data:[float])  -> [float]:
            var_mean=np.nanmean(data).round(2)
            var_median=np.nanmedian(data).round(2)
            var_s=np.var(data).round(1)
            var_n=len(data)
            val_=np.array(data)
            counts, bin_edges = np.histogram(val_[~np.isnan(val_)], bins=bins,density=True)
            cdf = np.cumsum(counts)
            tmp=max(cdf)
            cdf=cdf/float(tmp)
            return var_mean,var_median,var_s,var_n,bin_edges,cdf
               
        if nvar==1:
            var_mean,var_median,var_s,var_n,bin_edges,cdf=calc(data_var)
            if label:
                label_=f'{label} : n={var_n}, mean= {var_mean}, median= {var_median}, $\sigma^{2}$={var_s}'
                plt.plot(bin_edges[1:], cdf,color=colors, linewidth=linewidth,
                    label=label_)                
            else:
                plt.plot(bin_edges[1:], cdf,color=colors, linewidth=linewidth)

        else:    
            # Loop over variables
            for i in range (nvar):
                data = data_var[i]
                var_mean,var_median,var_s,var_n,bin_edges,cdf=calc(data)
                label_=f'{label[i]} : n={var_n}, mean= {var_mean}, median= {var_median}, $\sigma^{2}$={var_s}'
                plt.plot(bin_edges[1:], cdf,color=colors[i], linewidth=linewidth,
                        label=label_)
         
        plt.xlabel(xlabel,fontsize=x_ftze, labelpad=6)
        plt.ylabel(ylabel,fontsize=y_ftze)
        plt.title(title,fontsize=tit_ftze)
        if label:
            plt.legend(loc=loc,fontsize=leg_ftze,markerscale=1.2)
        
        ax1.grid(linewidth='0.2')
        plt.xlim(xlim) 
        plt.ylim(ylim)         

    ######################################################################### 
            
    def CrossPlot (x:list,y:list,title:str,xlabl:str,ylabl:str,loc:int,
                   xlimt:list,ylimt:list,axt=None,scale: float=0.8,alpha: float=0.6,
                   markersize: float=6,marker: str='ro', fit_line: bool=False, 
                   font: int=5) -> None:
        """
        Cross plto between two variables
         
        """
        ax1 = axt or plt.axes()
        font = {'size'   : font }
        plt.rc('font', **font) 

        x=np.array(x)
        y=np.array(y)    
        no_nan=np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
        Mean_x=np.mean(x)
        SD_x=np.sqrt(np.var(x)) 
        #
        n_x=len(x)
        n_y=len(y)
        Mean_y=np.mean(y)
        SD_y=np.sqrt(np.var(y)) 
        corr=np.corrcoef(x[no_nan],y[no_nan])
        n_=len(no_nan)
        #txt=r'$\rho_{x,y}=$%.2f'+'\n $n=$%.0f '
        #anchored_text = AnchoredText(txt %(corr[1,0], n_),borderpad=0, loc=loc,
        #                         prop={ 'size': font['size']*0.95, 'fontweight': 'bold'})  
        
        txt=r'$\rho_{x,y}}$=%.2f'+'\n $n$=%.0f \n $\mu_{x}$=%.0f \n $\sigma_{x}$=%.0f \n '
        txt+=' $\mu_{y}$=%.0f \n $\sigma_{y}$=%.0f'
        anchored_text = AnchoredText(txt %(corr[1,0], n_x,Mean_x,SD_x,Mean_y,SD_y), loc=4,
                                prop={ 'size': font['size']*1.1, 'fontweight': 'bold'})    
            
        ax1.add_artist(anchored_text)
        Lfunc1=np.polyfit(x,y,1)
        vEst=Lfunc1[0]*x+Lfunc1[1]    
        try:
            title
        except NameError:
            pass  # do nothing! 
        else:
            plt.title(title,fontsize=font['size']*(scale))   
    #
        try:
            xlabl
        except NameError:
            pass  # do nothing! 
        else:
            plt.xlabel(xlabl,fontsize=font['size']*scale)            
    #
        try:
            ylabl
        except NameError:
            pass  # do nothing! 
        else:
            plt.ylabel(ylabl,fontsize=font['size']*scale)        
            
        try:
            xlimt
        except NameError:
            pass  # do nothing! 
        else:
            plt.xlim(xlimt)   
    #        
        try:
            ylimt
        except NameError:
            pass  # do nothing! 
        else:
            plt.ylim(ylimt)   
          
        plt.plot(x,y,marker,markersize=markersize,alpha=alpha)  
        if fit_line:
            ax1.plot(x, vEst,'k-',linewidth=2)   
        ax1.grid(linewidth='0.1') 
        plt.xticks(fontsize=font['size']*0.85)    
        plt.yticks(fontsize=font['size']*0.85)    
        
    #########################################################################         
        
class Correlation_plot:
    def corr_mat(df: pd.DataFrame, title: str, corr_val_font: float=False, y_l: list=1.2,axt: plt.Axes=None,
                titlefontsize: int=10, xyfontsize: int=6, xy_title: list=[-22,1.2],
                vlim=[-0.8,0.8]) -> [float]:
        
        """Plot correlation matrix between features"""
        ax = axt or plt.axes()
        colmn=list(df.columns)
        corr=df.corr().values
        corr_array=[]
        for i in range(len(colmn)):
            for j in range(len(colmn)):
                c=corr[j,i]
                if (corr_val_font):
                        ax.text(j, i, str(round(c,2)), va='center', ha='center',fontsize=corr_val_font)
                if i>j:
                    corr_array.append(c)

        im =ax.matshow(corr, cmap='jet', interpolation='nearest',vmin=vlim[0], vmax=vlim[1])
        
        cbaxes = fig.add_axes([0.92, 0.23, 0.03, 0.50]) 
        cbar =fig.colorbar(im,cax=cbaxes,shrink=0.5,label='Correlation Coefficient')
        cbar.ax.tick_params(labelsize=10) 
        
        ax.set_xticks(np.arange(len(corr)))
        ax.set_xticklabels(colmn,fontsize=xyfontsize, rotation=90)
        ax.set_yticks(np.arange(len(corr)))
        ax.set_yticklabels(colmn,fontsize=xyfontsize)
        ax.grid(color='k', linestyle='-', linewidth=0.025)
        plt.text(xy_title[0],xy_title[1],title, 
                 fontsize=titlefontsize,bbox=dict(facecolor='white', alpha=0.2))
        return corr_array
        plt.show()
        
        
    #########################  
    
    def corr_bar(corr: list, clmns: str,title: str, select: bool= False
                ,yfontsize: float=4.6, xlim: list=[-0.5,0.5], ymax_vert_lin: float= False) -> None:
        
        """Plot correlation bar with target"""
        
        r_ = pd.DataFrame( { 'coef': corr, 'positive': corr>=0  }, index = clmns )
        r_ = r_.sort_values(by=['coef'])
        if (select):
            selected_features=abs(r_['coef'])[:select].index
            r_=r_[r_.index.isin(selected_features)]
    
        r_['coef'].plot(kind='barh',edgecolor='black',linewidth=0.8
                        , color=r_['positive'].map({True: 'r', False: 'b'}))
        plt.xlabel('Correlation Coefficient',fontsize=6)
        if (ymax_vert_lin): plt.vlines(x=0,ymin=-0.5, ymax=ymax_vert_lin, color = 'k',linewidth=1.2)
        plt.yticks(np.arange(len(r_.index)), r_.index,rotation=0,fontsize=yfontsize,x=0.01)
        plt.title(title)
        plt.xlim((xlim[0], xlim[1])) 
        ax1 = plt.gca()
        ax1.xaxis.grid(color='k', linestyle='-', linewidth=0.1)
        ax1.yaxis.grid(color='k', linestyle='-', linewidth=0.1)
        plt.show()   






