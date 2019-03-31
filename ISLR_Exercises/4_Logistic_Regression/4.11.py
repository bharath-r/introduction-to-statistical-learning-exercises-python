import numpy as np
import matplotlib.pyplot as plt

def Power():
    return(2**3)
    
def Power2(x, a):
    return(x**a)
    
def Power3(x, a):
    result= x**a
    return(result)
    
def Plot(log= ''):
    x= np.arange(1, 10)
    y= Power3(x, 2)
    
    fig, ax= plt.subplots()
    
    ax.set_xlabel('x')
    ax.set_ylabel('y=x^2')
    ax.set_title('Power3()')
    
    if log == 'x':
        ax.set_xscale('log')
        ax.set_xlabel('log(x)')
        
    if log == 'x':
        ax.set_yscale('log')
        ax.set_ylabel('log(y = x^2)')
        
    if log == 'xy':
        ax.set_xscale('log')
        ax.set_xlabel('log(x)')
        ax.set_yscale('log')
        ax.set_ylabel('log(y = x^2)')
        
    ax.plot(x, y)
    
Plot(log = 'xy')
        

def PlotPower(start,end,power,log=''):
    x = np.arange(start,end)
    y = np.power(x,end)

    #create plot
    fig, ax = plt.subplots()

    #config plot
    ax.set_xlabel('x')
    ax.set_ylabel('y=x^2')
    ax.set_title('PlotPower()')

    #change scale according to axis
    if log == 'x':
        ax.set_xscale('log')
        ax.set_xlabel('log(x)')
    if log == 'y':
        ax.set_yscale('log')
        ax.set_ylabel('log(y=x^2)')
    if log == 'xy':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('log(x)')
        ax.set_ylabel('log(y=x^2)')

    #draw plot
    ax.plot(x, y)
    
PlotPower(1, 10, 3)