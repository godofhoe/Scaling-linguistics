# -*- coding: utf-8 -*-
'''
@author  gmking

This module is used to run all file in the document "input" once instead of running case by case.
The function here won't show plots. 
If you want to see every picture of your txt, you should use Run_case_by_case.ipynb and DONOT import this module.
If you discover that some formulas on your plots miss their positions, you can adopt them case by case in Statistics.ipynb.
'''

import random 
import bisect 
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .count import *
from .count_col import *
from .Curve_Fitting_MLE import *
from scipy.optimize import curve_fit

def which_plot(name, V, H, x = 'H', SP = 'T', FORMAT = 'png', max_range = 50, shift = 'N', Path = ''):
    '''check ratio of geometric sequence {Hn} or {Vn}

       parameters:
    1. name: str
       "XXX" (your file name without filename extension)

    2. V, H: list or np.array
       V and H are the coordinates of the sequence {V} and {H}.
       You should get these two from 
             V, H = geometric_sequence(word, syl)
       where geometric_sequence(word, syl) is the function of count.py

    3. max_range: number
        the number of elements in the sequence you want to know

    4. x: 'H' or 'V'
        you can chose the sequence you want (H/V)

    5. FORMAT: png, pdf, ps, eps and svg
    '''
    
    if x == 'H':
        if len(H) < max_range + 4:
            max_range = len(H) - 5
        r = np.zeros(max_range - 2)
        
        if shift == 'T':
            def r_H_shift(x_0, h):
                h = np.array(h)
                r_shift = (h[2:max_range] - x_0)/ (h[1:max_range - 1] - x_0)
                std = np.sqrt(np.mean((r_shift - r_shift.mean())**2))
                return std
            
            #To get the value minimize std of r_shift, we don't use minimize() here because 
            #there are some problems in its algorithm. Instead, we use the Brute-force search
            find_r = []
            for x_0 in range(0, int(H[0]/2)):
                find_r.append(r_H_shift(x_0, H))
            SHIFT = find_r.index(min(find_r)) + 1
            h = np.array(H)
            r = (h[2:max_range] - SHIFT)/ (h[1:max_range -1] - SHIFT)
        
        elif shift != 'T': 
            SHIFT = 0
            for i in range(1, max_range - 1): #H[0]=H_1, H[1]=H_2
                r[i - 1] = H[i + 1]/ H[i]
                
        r_position = [i + 2 for i in range(len(r))] #we start from H_2
        STD = round(np.std(r), 3)
        MEAN = round(np.mean(r), 3)
        fig, ax = plt.subplots()
        ax.errorbar(r_position, r, yerr = STD) #plot errorbar         
        plt.text(max_range / 20, 0.3, '$r_H=%.3f\pm %.3f$' % (MEAN, STD), fontsize=35)        
        
        plt.title(name, size = 20)
        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.xlabel('index n', size = 20)
        plt.ylabel('$r_H$', size = 20)
        plt.ylim([0, max(r) + 0.1])
        plt.plot(r_position, r, 'ro')        
        if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):
            fig.savefig(Path + 'H of ' + name + '.' + FORMAT, dpi = 1000, format = FORMAT)
        plt.close()
    elif x == 'V':
        if len(V) < max_range + 4:
            max_range = len(V) - 5
        r = np.zeros(max_range - 2)
        
        if shift == 'T':
            def r_V_shift(x_0, v):
                v = np.array(v)
                r_shift = (v[2:max_range] - x_0)/ (v[1:max_range - 1] - x_0)
                std = np.sqrt(np.mean((r_shift - r_shift.mean())**2))
                return std
            
            #To get the value minimize std of r_shift, we don't use minimize() here because 
            #there are some problems in its algorithm. Instead, we use the Brute-force search
            find_r = []
            for x_0 in range(0, int(V[0]/2)):
                find_r.append(r_V_shift(x_0, V))
            SHIFT = find_r.index(min(find_r)) + 1
            v = np.array(V)
            r = (v[2:max_range] - SHIFT)/ (v[1:max_range -1] - SHIFT)
        
        elif shift != 'T': 
            SHIFT = 0
            for i in range(1, max_range - 1): #V[0]=V_1, V[1]=V_2
                print(V[i], V[i+1])
                r[i - 1] = V[i + 1] / V[i]                
        
        r_position = [i + 2 for i in range(len(r))] #we start from V_2
        STD = round(np.std(r), 3)
        MEAN = round(np.mean(r), 3)
        fig, ax = plt.subplots()
        ax.errorbar(r_position, r, yerr = STD) #plot errorbar
        plt.text(max_range / 20, 0.3, '$r_V=%.3f\pm %.3f$' % (MEAN, STD), fontsize=35)
        
        plt.title(name, size = 20)
        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.xlabel('index n', size = 20)
        plt.ylabel('$r_V$', size = 20)
        plt.ylim([0, max(r) + 0.1])
        plt.plot(r_position, r, 'ro')        
        if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):
            fig.savefig(Path + 'V of ' + name + '.' + FORMAT, dpi = 500, format = FORMAT)
        plt.close()
    else:
        print('please chose x = \'H\' or \'V\'')

def RRD_plot(big, word, syl, longest, name, V, H, need_line = 'Y', number_of_lines = 4, Color = '#ff0000', SP = 'T', FORMAT = 'png', Path = ''):
    '''draw the RRD plot and auxiliary lines
    
    Controllable parameters:
    --- 
    need_line: string
        If you don't want the auxiliary lines, change Y into other thing.

    number_of_lines: number
        How many auxiliary lines you need ? (both horizontal and vertical lines)
    Color: colorcode
    SP : str
        If you don't want to save picture, just assign SP != 'T'.
    FORMAT: string
        The format of your RRD plot. Most backends support png, pdf, ps, eps and svg.
    
    
    Fixed parameters:
    ---(please don't change them)
    big, word, syl, longest: pandas.DataFrame
        the output of the function info()
    H, V: ndarray
        the output of the function check_const_ratio
           
    output:
        show a RRD plot
    
    '''
    if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'): 
        fig, ax = plt.subplots()   
        if need_line == 'Y':

            Slice_number = 50 #this value decide the number of points on horizontal and vertical lines
            x_range = np.linspace(0, len(word), Slice_number)
            y_range = np.linspace(0, len(syl), Slice_number)


            for i in range(number_of_lines):
                x_const = [V[i] for j in range(Slice_number)]#x_const =[V[i], V[i], ..., V[i]], Slice_number elements
                y_const = [H[i] for j in range(Slice_number)] #y_const =[H[i], H[i], ..., H[i]], Slice_number elements
                plt.plot(x_range, y_const) #plot y=H[i] on RRD plot
                plt.plot(x_const, y_range) #plot x=V[i] on RRD plot   


        for i in range(longest): #draw 0th_syl ~ longest_th_syl on RRD plot
            str_position = [i + 1 for i in range(len(big[str(i) + "th_syl_rank"]))] #position starting form 1 not 0
            plt.plot(str_position, big[str(i) + "th_syl_rank"], 'o', markersize=3, color = Color)

        plt.xlabel('word', size = 15)
        plt.ylabel('syllable', size = 15)
        plt.xlim([0, max(word['wordRank'])*11/10])
        plt.ylim([0, max(syl['sylRank'])*17/15])
        plt.title(name, size = 20)
    
        fig.savefig(Path + 'RRD of ' + name + '.' + FORMAT, dpi=400, format = FORMAT) #adjust dpi if you want the figure more clear
        plt.close()
    else:
        print('no RRD plot.')
    
def FRD_plot(name, word, syl, x_pos = 2, y_pos = 10, SP = 'T', FORMAT = 'png', Path = ''):
    '''draw FRD plot of words and syllables

       parameters:
    0. name: str
       "XXX" (your file name without filename extension)

    1. word, syl: pd.daframe
       output of function info() or N_gram_info() in count.py
       you should get them from
       big, syl, word, longest = info(filename, encode)

    2. x_pos, y_pos : number
       (x_position, y_position) of your formula on FRD plot

    3. SP: str
       If you don't want to save picture, just assign SP != 'T'.

    5. FORMAT: str
       'png', 'pdf', 'ps', 'eps' and 'svg'
    '''
    if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):
        wf = word['wordFreq']
        cf = syl['sylFreq']
        max_wf = wf[0]
        max_cf = cf[0]

        #use MLE to get the fitting parameter, detial read: Curve_Fitting_MLE
        #-----------------------------------------
        T = ([],[])
        for i in word['wordRank']:
            T[0].append(i)
        for i in wf:
            T[1].append(i)
        #T = ([wordRank], [wordFreq])
        Y = Two_to_One(T)
        res = minimize(L_Zipf, 1.2, Y, method = 'SLSQP')
        s = res['x']
        t = [int(min(T[0])), int(max(T[0])), s]
        C = 1 / incomplete_harmonic(t)
        fig, ax = plt.subplots()
        plt.xlabel('rank', size = 20)
        plt.ylabel('frequency', size = 20)
        plt.title(name, size = 20)

        xdata = np.linspace(min(T[0]), max(T[0]), num = (max(T[0]) - min(T[0]))*10)
        theo = Zipf_law(xdata, s, C) #Notice theo is normalized, i.e, the probability density
        N = sum(T[1])
        theo = [N * i for i in theo] #change theo from probability density to real frequency

        #plt.text(x_position, y_position)
        if (x_pos, y_pos) == (0,0):
            x_mid = 1.2
            y_min = 0.2
            plt.text(x_mid, y_min,'$%.3fx^{-%.2f}$'%(C, s), fontsize=40) #write formula on the plot
        else:
            plt.text(x_pos, y_pos,'$%.3fx^{-%.2f}$'%(C, s), fontsize=40) #write formula on the plot
            
        plt.plot(xdata, theo, 'g-')
        #-----------------------------------------
        plt.ylim([0.1, 10*max(max_wf, max_cf)])
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(wf, 'ro', label = 'word', markersize=4)
        plt.plot(cf, 'x', label = 'syl', markersize=6)
        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.legend(loc = 'best', prop={'size': 20})
        fig.savefig(Path + 'FRD of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
        plt.close()
    else:
        print('no FRD plot.')

def Col_plot(name, syl, x_pos = 10, y_pos = 0, SP = 'T', FORMAT = 'png', Path = ''):
    '''draw collocation-rank plot 

       parameters:
    0. name: str
       "XXX" (your file name without filename extension)

    1. syl: pd.daframe
       output of function info() or N_gram_info() in count.py
       you should get them from
       big, syl, word, longest = info(filename, encode)

    2. x_pos, y_pos : number
       (x_position, y_position) of your formula on FRD plot

    3. SP: str
       If you don't want to save picture, just assign SP != 'T'.

    5. FORMAT: str
       'png', 'pdf', 'ps', 'eps' and 'svg'
    '''
    if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):
        Syl = syl.sort_values(by = '#collocations', ascending=False)
        reSyl = Syl.reset_index()

        #use OLS to get the fitting parameter
        #-----------------------------------------
        def col(y, a, b):
            return (a * np.log(y) + b) ** 2

        popt, pcov = curve_fit(col, syl['sylRank'], reSyl['#collocations'])
        #popt is the optimal values for the parameters (a,b)
        theo = col(syl['sylRank'], *popt)
        fig, ax = plt.subplots()
        plt.plot(syl['sylRank'], theo, 'g--')
    
        a = 7   #auto positioning, m = min(syl['sylRank']) = 1 always
        b = 1   #auto positioning, M = max(syl['sylRank'])
        xmid = max(syl['sylRank'])**(b/(a+b))  #exp([a*log(m)+b*log(M)]/[a+b]) = m^(a/[a+b]) * M^(b/[a+b])    
        ytop = max(reSyl['#collocations'])*5/6

        #the following code deal with significant figures of fitting parameters
        #the tutor of significant figures: https://www.usna.edu/ChemDept/_files/documents/manual/apdxB.pdf
        #-----------------------------------------
        col_dig = len(str(max(reSyl['#collocations'])))
        yp_dig = len(str(max(syl['sylRank']))) #ln(y') will have yp_dig +1 digits (yp_dig significant figures)
        a_dig = min(col_dig, yp_dig +1) #significant figures of parameter a
        b_dig = col_dig #significant figures of parameter b

        # the fomat string is #.?g, where ? = significant figures
        # detail of the fomat string: https://bugs.python.org/issue32790
        # https://docs.python.org/3/tutorial/floatingpoint.html
        A = format(popt[0], '#.%dg' % a_dig)  # give a_dig significant digits
        B = format(popt[1], '#.%dg' % b_dig)  # give b_dig significant digits
        if 'e' in A: #make scientific notation more beautiful
            A = A.split('e')[0] + '\\times 10^{' + str(int(A.split('e')[1])) + '}'
        if A[-1] == '.':
            A = A[:-1]
        if 'e' in B: #make scientific notation more beautiful
            B = B.split('e')[0] + '\\times 10^{' + str(int(B.split('e')[1])) + '}'
        if B[-1] == '.':
            B = B[:-1]

        equation_string = '$(%s\ln{y\prime}+%s)^2$' % (A, B)    

        if x_pos != 0 and y_pos != 0:
            plt.text(x_pos, y_pos, equation_string, fontsize=30)
        elif x_pos != 0 and y_pos == 0:
            plt.text(x_pos, ytop, equation_string, fontsize=30)
        elif x_pos == 0 and y_pos != 0:
            plt.text(xmid, y_pos, equation_string, fontsize=30)
        else:
            plt.text(xmid, ytop, equation_string, fontsize=30)
        #-----------------------------------------

        plt.plot(reSyl['#collocations'], 'ro', label = 'syl', markersize=4)
        plt.xlabel('rank of syls($y\prime$)', size = 20)
        plt.xscale('log')
        plt.ylabel('collocations', size = 20)
        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.title(name, fontsize = 20) 
        fig.savefig(Path + 'collocation_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
        plt.close()
    else:
        print('no collocation-rank plot.')
        
def Link_plot(name, word, x_pos = 20, y_pos = 0, SP = 'T', FORMAT = 'png', Path = ''):
    '''draw link-rank plot  

       parameters:
    0. name: str
       "XXX" (your file name without filename extension)

    1. syl: pd.daframe
       output of function info() or N_gram_info() in count.py
       you should get them from
       big, syl, word, longest = info(filename, encode)

    2. x_pos, y_pos : number
       (x_position, y_position) of your formula on FRD plot

    3. SP: str
       If you don't want to save picture, just assign SP != 'T'.

    5. FORMAT: str
       'png', 'pdf', 'ps', 'eps' and 'svg'
    '''
    if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):
        Word = word.sort_values(by='#links', ascending=False)
        reWord = Word.reset_index()

        #use OLS to get the fitting parameter
        #-----------------------------------------
        def link(x, a, b):
            return (a * np.log(x) + b)

        popt, pcov = curve_fit(link, word['wordRank'], reWord['#links'])
        #popt is the optimal values for the parameters (a,b)
        theo = link(word['wordRank'], *popt)
        fig, ax = plt.subplots()
        plt.plot(word['wordRank'], theo, 'g--')

        a = 12   #auto positioning, m = min(word['wordRank'])
        b = 1   #auto positioning, M = max(word['wordRank'])
        xmid = max(word['wordRank']) ** (b/(a+b))  #exp([a*log(m)+b*log(M)]/[a+b]) = m^(a/[a+b]) * M^(b/[a+b])
        ytop = max(reWord['#links']) 

        #the following code deal with significant figures of fitting parameters
        #the tutor of significant figures: https://www.usna.edu/ChemDept/_files/documents/manual/apdxB.pdf
        #-----------------------------------------
        link_dig = len(str(max(reWord['#links'])))
        xp_dig = len(str(max(word['wordRank']))) #ln(x') will have xp_dig +1 digits (xp_dig significant figures)
        a_dig = min(link_dig, xp_dig +1) #significant figures of parameter a
        b_dig = link_dig #significant figures of parameter b

        # the fomat string is #.?g, where ? = significant figures
        # detail of the fomat string: https://bugs.python.org/issue32790
        # https://docs.python.org/3/tutorial/floatingpoint.html
        A = format(popt[0], '#.%dg' % a_dig)  # give a_dig significant digits
        B = format(popt[1], '#.%dg' % b_dig)  # give b_dig significant digits
        if 'e' in A: #make scientific notation more beautiful
            A = A.split('e')[0] + '\\times 10^{' + str(int(A.split('e')[1])) + '}'
        if A[-1] == '.':
            A = A[:-1]
        if 'e' in B: #make scientific notation more beautiful
            B = B.split('e')[0] + '\\times 10^{' + str(int(B.split('e')[1])) + '}'
        if B[-1] == '.':
            B = B[:-1]

        equation_string = '$%s\ln{x\prime}+%s$' % (A, B)

        if x_pos != 0 and y_pos != 0:
            plt.text(x_pos, y_pos, equation_string, fontsize=30) 
        elif x_pos != 0 and y_pos == 0:
            plt.text(x_pos, ytop, equation_string, fontsize=30) 
        elif x_pos == 0 and y_pos != 0:
            plt.text(xmid, y_pos, equation_string, fontsize=30) 
        else:
            plt.text(xmid, ytop, equation_string, fontsize=30) 

        #-----------------------------------------

        plt.plot(reWord['#links'], 'ro', label = 'word', markersize = 4)
        plt.xlabel('rank of words($x\prime$)', size = 20)
        plt.xscale('log')
        plt.ylabel('links', size = 20)
        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.title(name, size = 20)
        fig.savefig(Path + 'links_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
        plt.close()
    else:
        print('no Link-rank plot.')
        

def Plot_f(n, V, H, big, longest, toler = 50, avg_N = 50):
    '''
    ---input
    n: integer, this function will select points on scaling line from f_1 to f_n
    toler: float, tolerance. Increase this value will speed up the minimization process but decline in performance.
    avg_N: number of segmentation (N block) that used to  doing moving avgerage.
    
    ---return
    f: set, {f_1, f_2,...,f_n}, where f_k = (x_avg, y_avg), x/y_avg is points after moving average
    flu: set, {flu_1, flu_2,...,flu_n}, where flu_k = (lupx, lupy), lupx/y is points on scaling line
    '''
    f = {}
    flu = {}
    #pick up points on scaling line
    for M in range(1, n+1):
        (M, N) = (M, 1)
        points = chose_point(M, N, V, H, big, longest) #see count_col.py
        px, py = sort_point(M, N, points) #see count_col.py
        if M == 1:
            luptx, lupty = px, py
        else:
            luptx, lupty = DENOISE(M, N, V, H, big, longest, toler) #see count_col.py
        flu['f' + str(M)] = (luptx, lupty)
        #moving average
        Range = [0.25*V[0], V[0]]
        x_avg, y_avg = moving_avg(luptx, lupty, Range, avg_N) #see count_col.py
        f['f' + str(M)] = (x_avg, y_avg)
    return f, flu

def rf(name, SP, FORMAT, f, Path = ''):
    '''plot r_f of your data
    
    ------paras
    name: str
       name of your r_f plot
    SP: str, should be 'T' or others
       If you don't want to save picture, just assign SP != 'T'.
    FORMAT: str
       format of your picture
    f: set, contain points on f1 ~ fn after moving average
       output of plot_f(n, V, H) #suggestion: use flu(non-average data), not f(average data)
       
    ------output
    x: x coordinate
    y: f_i/f_(i+1) for all i
    STD: standard error for f_i/f_(i+1)    
    if SP = 'T': a picture
    '''
    if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):
        avg_N = max([len(f[i][0]) for i in f])
        lf = len(f)
        y = {}
        x = {}
        STD = {} #STD of f_i/f_(i+1)
        weight = {} #number of data of f_i/f_(i+1)
        r = {} #average ratio of f_i/f_(i+1)
        for n in range(1, lf):
            fn1 = 'f' + str(n+1)
            fn = 'f' + str(n)
            y[fn1 + '/' + fn] = [f[fn1][1][j]/f[fn][1][j] for j in range(avg_N)]
            x[fn1 + '/' + fn] = [0.5*f[fn1][0][j] + 0.5*f[fn][0][j] for j in range(avg_N)]

            y_n = [i for i in y[fn1 + '/' + fn] if i==i] # this is y without NAN
            STD[fn1 + '/' + fn] = round(np.std(y_n), 3)
            weight[fn1 + '/' + fn] = len(y_n)
            r[fn1 + '/' + fn] = round(np.mean(y_n), 3)

        fig, ax = plt.subplots()
        #calculate SP value excluding f2/f1
        error = {}
        for i in STD:
            if i != 'f2/f1':
                error[i] = STD[i]
        del weight['f2/f1'], r['f2/f1']      

        P = {}
        for i in x:
            px = x[i]
            py = y[i]
            std = STD[i]
            if i != 'f2/f1':
                P[i] = weight[i]/len(py)
                if P[i] < 0.6:
                    print('P < 0.6: %s, %f' % (i, P[i]))
                    del weight[i], r[i], P[i] #we don't count such r_f but still draw it on r_f plot
                    del f[i.split('/')[0]] #ex: P['f4/f3'] < 0.6, del data['f4']
                    
            ax.errorbar(px, py, yerr = std) #plot errorbar
            plt.plot(px, py,'o', markersize = '4', label = i)
            plt.legend(loc = 'best', prop = {'size': 15})
        
        tot = sum([weight[w] for w in weight])
        R = sum([weight[i]*r[i]/tot for i in weight])
        ERROR = (sum([weight[i]*error[i]**2/tot for i in weight]))**0.5
        P_value = np.mean([P[i] for i in P])
        S_value = 1 - ERROR/R

        xmin, xmax = plt.xlim([0,None])
        ymin, ymax = plt.ylim([0,None])
        plt.text(xmax*0.98, ymax*0.5, '$r_f=%.3f\pm %.3f$' % (R,ERROR), fontsize=35, verticalalignment='bottom', horizontalalignment='right')
        plt.text(xmax*0.85, ymax*0.35, '$S=%.3f$' % (S_value), fontsize=35, verticalalignment='bottom', horizontalalignment='right')
        plt.text(xmax*0.85, ymax*0.2, '$P=%.3f$' % (P_value), fontsize=35, verticalalignment='bottom', horizontalalignment='right')
        plt.text(xmax*0.85, ymax*0.05, '$SP=%.3f$' % (S_value*P_value), fontsize=35, verticalalignment='bottom', horizontalalignment='right')
        plt.xlabel('$x$', size = 20)
        plt.ylabel('$r_f(x)$', size = 20)
        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        #https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html
        ax.ticklabel_format(axis='x', style='sci',scilimits=(0,3))
        #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.title(name, size = 20)
        fig.savefig(Path + 'rf of ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
        plt.close()
    else:
        print('no ratio of scaling lines plot.')
    return (R, ERROR)

def Scaling_fit(data, Rf, V, H, name, SP, FORMAT, Path = ''):
    '''find out best fitting curve for scaling lines
    use fun to be fitting model, select f1~fn to be basis of scaling function, after that find out the best basis and parameters
    by check deviation of different basis.
    ### Notice: we don't use f1 as basis, and exclude f1 when calculate deviation
    For instance, use f2 to be basis
    0. fitting f2 with fun
    1. f3 = Rf*f2, f4 = Rf^2*f2......
    2. tot_Dev['f2'] = sum((y_i - f_i)**2), where y_i is real data and f_i is fitting data
    3. check all possible and resonable basis, findout the smallest tot_Dev['fn'] 
    
    ------paras
    data: set of points on scaling line, it is output of 
          f, flu = plot_f(4, V, H, big, longest, toler = 50, avg_N = 50)
          these data should contain no nan. I don't suggest use moving average data
    Rf: ratio of scaling curve
        Instead of Rf, you can try RH
        Rf = rf(name, SP, FORMAT, f)
        Rf[0] = mean, Rf[1] = std
        RH = which_plot(x, SP, FORMAT, max_range, shift)
        RH[0] = mean, RH[1] = std, RH[2] = shift
        
    ------output
    1. a picture with best fitting curve
    2. fitting parameters
    
    '''
    def fun(x, q, s, t):
        '''theory of scaling curve                
        '''
        return q*Rf**(s*(x)**-t)    
    number = len(data) #number of scaling lines need fitting
    q0 = (2000, 3000, 0.94) #initial guess
    fit_para = {}
    tot_Dev = {}
    for fn in data:
        if fn != 'f1':           
            theo = {}
            Dev = {}
            #popt is the optimal values for the parameters (q, s, t)
            popt, pcov = curve_fit(fun, data[fn][0], data[fn][1], q0, bounds =(0, [np.inf,np.inf,1.5]))
            fit_para[fn] = (popt, pcov)   
            n = int(fn.split('f')[1]) #ex: fn = 'f2' then n = 2
            for N in range(2, number + 1):
                FN = 'f' + str(N)
                theo[FN] = fun(data[FN][0], *popt) * Rf**(N - n) #ex: n=2, FN=4 then theo['f3'] = fun*Rf^2
                dif = theo[FN] - data[FN][1]
                Dev[FN] = np.sum(np.square(dif))
            tot_Dev[fn] = np.sum([Dev[i] for i in Dev])
        
    best = min(tot_Dev, key = tot_Dev.get) #Get the key corresponding to the minimum value within a dictionary
    n = int(best.split('f')[1]) #ex: best = 'f2' then n = 2
    fig, ax = plt.subplots()
    for N in range(1, number + 1):
        FN = 'f' + str(N)
        theo[FN] = fun(data[FN][0], *fit_para[best][0]) * Rf**(N - n) #ex: n=2, N_F=1 then theo['f1] = Rf^(-1)* fun
        plt.plot(data[FN][0], data[FN][1], '.', markersize = '4', color ='#e9bf53')
        plt.plot(data[FN][0], theo[FN], 'o', markersize = '4')
    xm, xM = plt.xlim([0,V[0]*1.03])
    ym, yM = plt.ylim([0,H[0]*1.03])
    #-------------------------------------
    A = format(popt[0], '#.4g')  # give 4 significant digits
    if A[-1] == '.':
        A = A[:-1]
    B = format(popt[1], '#.4g')  # give 4 significant digits
    if B[-1] == '.':
        B = B[:-1]
    plt.text(0.05*xM, 0.05*yM, r'$f_%d=%s \times %.3f^{%s x^{-%.2f}}$' % (n, A, Rf, B, popt[2]), fontsize=24, color ='black')
    #-------------------------------------
    plt.xlabel('word', size = 15)
    plt.ylabel('syllable', size = 15)  
    plt.title(name, size = 20)
    
    #the following part is used to calculate fitting score, base on an empirical truth that good fitting use less data
    dx_min = {i:min(data[i][0]) for i in data} #find minima x in data
    if (V[0] - dx_min[min(dx_min)]) < 0.75*V[0]:
        score = 1
    elif V[0] > (V[0] - dx_min[min(dx_min)]) >= 0.75*V[0]:
        score = (0.75*V[0])/(V[0] - dx_min[min(dx_min)])
    else:
        score = 0.5
        
    plt.text(0.05*xM, 0.16*yM, 'fitting score: %.3f' % score, fontsize=24, color ='black')
    
    if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):
        fig.savefig(Path + 'fitting ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
    plt.close()
    return fit_para[best]

def fit_with_cut(data, Rf, V, H, name, SP, FORMAT, Path = ''):
    '''
    fit data bigger than 0.25*V[0] to rise accuracy of fitting
    if 0.25*V[0] is not small enough, lowering the low bound of data automatically
    '''
    data_range = [0.25 - i*0.01 for i in range(26)]
    check = 0
    for dr in data_range:
        try:
            D = {}
            for fn in data:
                b = [[],[]]                
                for i in range(len(data[fn][0])):        
                    if data[fn][0][i] >= dr*V[0]:
                        b[0].append(data[fn][0][i])
                        b[1].append(data[fn][1][i])
                D[fn] = (b[0], b[1])
            fit_para = Scaling_fit(D, Rf, V, H, name, SP, FORMAT, Path)
            print('fitting range = [%d, %d]' % (dr*V[0], V[0]))
            check = 1
            return fit_para
            break
        except RuntimeError:
            pass
    if check == 0 :
        print('Can not find best parameters in data range.')