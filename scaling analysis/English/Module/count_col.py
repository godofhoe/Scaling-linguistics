# -*- coding: utf-8 -*-
"""
Spyder Editor

@author  gmking, shan

This module is use to calculate the collocations and links of Chinese.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit

def count_col(pdframe1, pdframe2, feature1 = "word", feature2 = "syl", feature3 = "sylFreq"):
    '''count the collocations of syllables and the links of words
    
    input:
    pdframe1, pdframe2 : word and syl with class pandas.DataFrame. This two args are from the function, info(file_name, encode = "UTF-8"),  
    in the module count.py.
    
    output:
    add a frame "#collocations" (numbers of collocations of syls) in pdframe2
    add a frame "#links" (numbers of links of words) in pdframe1
        
    '''
    
    word_array = pdframe1[feature1] #ex: word_array=['apple','coffee','elephant']
    syl_array = pdframe2[feature2] #ex: syl_array=['ap', 'ple', 'cof', 'fee', 'e', 'le', 'phant']
    
    #First, we calculate collocations
    
    collocation = {}
    for c in syl_array: 
        collocation[c] = 0
        for w in word_array:
            t = w.split('-')
            if c in set(t): #ex: 'A' in 'AB', but 'A' not in 'BC'
                collocation[c] += 1

    syl_num_collocations_array = np.array([], dtype = 'int16' )
    
    for i in range(len(pdframe2)):
        syl_num_collocations_array = np.append(syl_num_collocations_array, collocation[syl_array[i]])
    
    #add a frame "#collocations" (numbers of collocations of syls) to syl
    pdframe2['#collocations'] = syl_num_collocations_array 
        
    #Second, we use collocation to calculate links
    
    
    link = {}
    for w in word_array:
        link[w] = 0
        t = w.split('-')
        for c in set(t):
            #If we don't use set(w) here, the links will be overcount. 
            #ex: link('AA') = collocation('A') but not 2*collocation('A')
            link[w] += collocation[c]
    
    link_num_array = np.array([], dtype = 'int16')
    
    for i in range(len(pdframe1)):
        link_num_array = np.append(link_num_array , link[word_array[i]])
    
    #add a frame "#links" (numbers of links of words) to word
    pdframe1['#links'] = link_num_array 
    

    return None

def chose_point(m, n, V, H, big, longest):
    '''chose the points in (m, n), (m+1, n+1), ... blocks, where m > n.
    return: points = [[(x_m,y_n),...], [(x_(m+1), y_(n+1)),...], ...]
    
    the position of (m, n) block is the same as element (m*n) in matrix (row m and column n)
    
    '''
    points = [[] for j in range(min(len(V) - n, len(H) - m))]
    for j in range(min(len(V) - n, len(H) - m)):
        #https://thispointer.com/python-pandas-select-rows-in-dataframe-by-conditions-on-multiple-columns/
        V_word = big.loc[(big['wordRank'] <= V[j+n-1]) & (big['wordRank'] > V[j+n])]
        for i in range(longest): #draw 0th_syl ~ (longest-1)_th_syl
            H_syl = V_word[(V_word[str(i) + "th_syl_rank"] <= H[j+m-1]) & (V_word[str(i) + "th_syl_rank"] > H[j+m])]
            ind = H_syl.index #if I don't use index here, the content of points will be series not value
            for k in ind:
                points[j].append((H_syl.loc[k, 'wordRank'], H_syl.loc[k, str(i) + "th_syl_rank"]))
    return points

def sort_point(m, n, p):
    '''
    sort points in each block to make analysis easier
    '''
    for i in p:
        #sort list with key
        i.sort(key = lambda x: x[1], reverse = True)
    px = []
    py = []
    for i in range(5):
        if p[i] == []:
            print ('the (%d, %d) block have no points.' % (i+m, i+n))
        for j in p[i]:
            px.append(j[0])
            py.append(j[1])
    return px, py

def left_upper(m, n, V, H, big, longest):
    points = chose_point(m, n, V, H, big, longest)
    lup = [[] for i in range(len(points))]
    for i in range(len(points)):
        for p in points[i]:
            if p[1] > p[0]*(H[i+m-1] - H[i+m])/(V[i+n-1] - V[i+n]) + H[i+m-1] - V[i+n-1]*(H[i+m-1] - H[i+m])/(V[i+n-1] - V[i+n]):
                lup[i].append(p)
    return lup

def Tv(r, p):
    '''
    input:
    r: ndarray
    data points after denosing,where r = [rx1, rx2, ..., rxn, ry1, ry2, ..., ryn]
    p: ndarray
    data with noise, where p = [x1, x2, ..., xn, y1, y2, ..., yn]
    
    output:
    a function used to denoise
    '''
    def penalty(u, M):
        #This penalty function is used to lower the influence of outliers
        if abs(u) <= M:
            return u ** 2
        elif abs(u) > M:
            return M * (2 * abs(u) - M)        
    t = 0
    for i in range(len(p)):
        t = t + penalty(r[i] - p[i], 50)
    
    #see taxicab distance
    Lambda = 1 #regularziation parameters
    rl = int(len(r)/2)
    return sum(np.square(r[1:rl]-r[:rl - 1])) + sum(np.square(r[rl + 1:]-r[rl:-1])) + Lambda*t

def DENOISE(m, n, V, H, big, longest, toler):  
    lup = left_upper(m, n, V, H, big, longest)
    lupx, lupy = sort_point(m, n, lup)
    lupxy = np.array(lupx + lupy)  #lupxy = [x1, x2, ..., xn, y1, y2, ..., yn]
    RR = minimize(Tv, lupxy, args = lupxy, method='CG', tol = toler)
    luptx = RR.x[:int(len(RR.x)/2)]
    lupty = RR.x[int(len(RR.x)/2):]
    return luptx, lupty

def moving_avg(px, py, Range, N = 50):
    '''use moving average on (px,py), ruturn p_avg
    ---input:
    px, py: 1-D list
    Range: the x-range of data points
    N: number of segmentation, i.e. segment pi into N block. the moving period = (max(Range)-min(Range))/N
    
    ---return:
    p_avg: (px,py) after moving average
    '''
    period = (max(Range) - min(Range))/N
    px_avg = [[] for i in range(N)]
    py_avg = [[] for i in range(N)]
    p = [(px[i], py[i]) for i in range(len(px))]
    
    for j in p:
        #check every points
        for i in range(N):
            if (j[0] >= min(Range) + i*period) & (j[0] <= min(Range) + (i+1)*period):
                px_avg[i].append(j[0])
                py_avg[i].append(j[1])
                break    
    x_avg, y_avg = [], []
    for i in range(N):
        if (px_avg[i] == []) or (py_avg[i] == []):
            x_avg.append(float('nan'))
            y_avg.append(float('nan'))
        else:
            x_avg.append(np.mean(px_avg[i]))
            y_avg.append(np.mean(py_avg[i]))        
    return x_avg, y_avg

def plot_f(n, V, H, big, name, longest, toler = 50, avg_N = 50):
    '''
    ---input
    n: integer, this function will select points on scaling line from f_1 to f_n
    toler: float, tolerance. Increase this value will speed up the minimization process but decline in performance.
    avg_N: number of segmentation (N block) that used to  doing moving avgerage.
    
    ---return
    f: set, {f_1, f_2,...,f_n}, where f_k = (x_avg, y_avg), x/y_avg is points after moving average
    flu: set, {flu_1, flu_2,...,flu_n}, where flu_k = (lupx, lupy), lupx/y is points on scaling line
    '''
    #-----------------------------plot horizontal and vertical lines
    Slice_number = 50 #this value decide the number of points on horizontal and vertical lines
    number_of_lines = 4
    x_range = np.linspace(0, max(V), Slice_number)
    y_range = np.linspace(0, max(H), Slice_number)
        
        
    for i in range(number_of_lines):
        x_const = [V[i] for j in range(Slice_number)]#x_const =[V[i], V[i], ..., V[i]], Slice_number elements
        y_const = [H[i] for j in range(Slice_number)] #y_const =[H[i], H[i], ..., H[i]], Slice_number elements
        plt.plot(x_range, y_const) #plot y=H[i]
        plt.plot(x_const, y_range) #plot x=V[i]   
    #-----------------------------   
    f = {}
    flu = {}
    #plt.locator_params(axis='y', nbins=5)
    #pick up points on scaling line
    for M in range(1, n+1):
        (M, N) = (M, 1)
        points = chose_point(M, N, V, H, big, longest)
        px, py = sort_point(M, N, points)
        plt.plot(px, py,'o', markersize = '4')
        if M == 1:
            luptx, lupty = px, py
        else:
            luptx, lupty = DENOISE(M, N, V, H, big, longest, toler)
        flu['f' + str(M)] = (luptx, lupty)
        plt.plot(luptx, lupty,'.' ,markersize = '4', color = '#e9bf53')
        #moving average
        Range = [0.25*V[0], V[0]]
        x_avg, y_avg = moving_avg(luptx, lupty, Range, avg_N)
        f['f' + str(M)] = (x_avg, y_avg)
    plt.xlim([0,V[0]*1.03])
    plt.ylim([0,H[0]*1.03])
    plt.xlabel('word', size = 15)
    plt.ylabel('syllable', size = 15)  
    plt.title(name, size = 20)
    plt.show()
    return f, flu


def scaling_fit(data, Rf, V, H, name, SP, FORMAT):
    '''find out best fitting curve for scaling lines
    use fun to be fitting model, select f1~fn to be basis of scaling function, after that find out the best basis and parameters
    by check deviation of different basis.
    ### Notice: we don't use f1 as basis, and exclude f1 when calculate deviation
    For instance, use f2 to be basis
    0. fitting f2 with fun
    1. f3 = Rf*f2, f4 = Rf^2*f2......
    2. tot_Dev['f2'] = sum((y_i - f_i)**2), where y_i is real data and f_i is fitting data
    3. check all possible and resonable basis, findout the smallest tot_Dev['fn'] 
    4. calculate fitting score, base on an empirical truth that good fitting use less data
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
    fig, ax = plt.subplots()
    #plt.locator_params(axis='y', nbins=5)
    #-----------------------------plot horizontal and vertical lines
    Slice_number = 50 #this value decide the number of points on horizontal and vertical lines
    number_of_lines = 4
    x_range = np.linspace(0, max(V), Slice_number)
    y_range = np.linspace(0, max(H), Slice_number)
        
        
    for i in range(number_of_lines):
        x_const = [V[i] for j in range(Slice_number)]#x_const =[V[i], V[i], ..., V[i]], Slice_number elements
        y_const = [H[i] for j in range(Slice_number)] #y_const =[H[i], H[i], ..., H[i]], Slice_number elements
        plt.plot(x_range, y_const) #plot y=H[i]
        plt.plot(x_const, y_range) #plot x=V[i]   
    #-----------------------------  
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
            popt, pcov = curve_fit(fun, data[fn][0], data[fn][1], q0, bounds =(0, [np.inf, np.inf, 1.5]))
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
        fig.savefig('fitting ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
    plt.show()
    return fit_para[best]


