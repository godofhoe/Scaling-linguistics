# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:02:47 2016

@author: shan, gmking

This module is used to construct a dataframe with all statistical information we need.
The core function of this module is info(file_name, encode = "UTF-8")


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sys
from .zipfgen import ZipfGenerator #https://medium.com/pyladies-taiwan/python-%E7%9A%84-import-%E9%99%B7%E9%98%B1-3538e74f57e3
import random


def read_file(filename, encode = 'UTF-8'):
    """
    Read the text file with the given filename;
    return a list of the words of text in the file; ignore punctuations.
    also returns the longest word length in the file.
    
    paras:
    --------
    file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.

    encode : encoding of your txt
    """
    punctuation_set = set(u'''_—＄％＃＆:#$&!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
    ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
    々‖•·ˇˉ－―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
    ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
    num = 0
    word_list = []
    with open(filename, "r", encoding = encode) as file:
        for line in file:
            l = line.split()
            new_word = ''
            for word in l:
                for c in word:
                    if c not in punctuation_set:
                        new_word = new_word + c
                    if c in punctuation_set and len(new_word) != 0:
                        word_list.append(new_word)
                        new_word = ''
                if len(new_word) != 0: 
                    if len(new_word) > num:
                        num = len(new_word) #max number of characters in a word
                    word_list.append(new_word)
                    new_word = ''
                    
    if '\ufeff' in word_list:
        word_list.remove('\ufeff')
    
    print("read file successfully!")
    return word_list, num

def read_Ngram_file(filename, N, encode = 'UTF-8'):
    """
    Read the text file with the given filename;    return a list of the words of text in the file; ignore punctuations.
    
    paras:
    --------
    file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.
        
    N: int
      "N"-gram. 
      For example : a string, ABCDEFG (In Chinese, you don't know what's the 'words' of a string)
      in 2-gram = [AB, CD, EF, G];
      in 3-gram = [ABC, DEF, G];
      in 4-gram = [ABCD, EFG]
      
      two word compose a txt 'ABCDE EFGHI' (This case happended in English corpus)
      in 2-gram = [AB, CD, E, EF, GH, I];
      in 3-gram = [ABC, DE, EFG, HI];
      in 4-gram = [ABCD, E, EFGH, I]
    
    encode : encoding of your txt
    """
    punctuation_set = set(u'''_—＄％＃＆:#$&!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
    ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
    々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
    ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

    word_list = []
    with open(filename, "r", encoding = encode) as file:
        for line in file:
            new_word = ''
            for word in line:
                if word not in punctuation_set:
                    new_word += word
                if len(new_word) != 0 and (word in punctuation_set or len(new_word) == N):
                    word_list.append(new_word)
                    new_word = ''
    
    if '\ufeff' in word_list:
        word_list.remove('\ufeff')
                    
    print("read file successfully!")
    return word_list, N


def count_frequency(word_list):
    """
    Input: 
        word_list: list
            a list containing words or characters
    Return: 
        D: set
            a dictionary mapping words to frequency.
    """
    D = {}
    for new_word in word_list:
        if new_word in D:
            D[new_word] = D[new_word] + 1
        else:
            D[new_word] = 1
    return D   


def decide_seq_order(word_list):
    """
    Input:
        word_list: list
            a list containing words or characters
    Return: 
        D: set
            a dictionary mapping each word to its sequential number, which is decided by the order it 
            first appears in the word_list.
        another_list: list
            a list containg non-repetitive words, each in the order it first appears in word_list.
    """
    D = {}
    another_list = []
    for word in word_list:
        if word not in another_list:
            another_list.append(word)
    for num in range(len(another_list)):
        D[another_list[num]] = num + 1
    
    return D, another_list


def transfrom_wordlist_into_charlist(word_list):
    """Divide each words in the word_list into characters, order reserved.
    Input: a list containing words
    Return: a list containg char 
    """
    char_list = []
    for word in word_list:
        char_list.extend(list(word))
        
    return char_list


def produce_data_frame(word_list, word_freq, word_seq, varibleTitle):
    word_list = list(set(word_list))
    data = {}
    word_seq_list = []
    word_freq_list = []
    
    for word in word_list:
        word_freq_list.append(word_freq[word])
        word_seq_list.append(word_seq[word])
    
    first = varibleTitle 
    second = varibleTitle + "SeqOrder"
    third = varibleTitle + "Freq"
    forth = varibleTitle + "Rank"
    
    data[first] = word_list
    data[second] = word_seq_list
    data[third] = word_freq_list  
    
    dataFrame = pd.DataFrame(data)
    dataFrame = dataFrame.sort_values([third, second],ascending = [False,True])
    rank = np.array(list(range(1,len(dataFrame)+1))) 
    dataFrame[forth] = rank
    column_list = [first, third, forth, second]
    dataFrame = dataFrame[column_list]
    dataFrame = dataFrame.reset_index(drop=True)
    return dataFrame


def produce_wordRank_charRank_frame(pd_word,pd_char,longest):
    
    D = {}
    
    char_array = pd_char["char"]
    char_rank = {}
    
    for i in range(len(pd_char)):
        char_rank[char_array[i]] = i + 1 
    
    for i in range(longest):
        D[i] = []
    
    word_array = pd_word["word"]
    
    for word in word_array:
        for i in range(len(word)):
            D[i].append(int(char_rank[word[i]]))
        
        if len(word) < longest:
            for j in range(len(word),longest):
                D[j].append(np.nan)
    
    for k in range(longest):
        feature = str(k) + "th" + "_char_rank"
        pd_word[feature] = np.array(D[k])
    
    return pd_word  


def info(file_name, encode = "UTF-8"):
    '''This is the main program.
        
    paras:
    --------
    file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.
    
    encode : encoding of your txt
    
    
    return:
    --------
    data_frame: pd.dataframe
      a big data frame contain the information of words and its compositition
    pd_char: pd.dataframe
      a data frame contain the information of characters
    another_word: pd.dataframe
      a data frame contain the information of words
    longest_L: int
      the biggest length of single word.
    
    '''
    
    L, longest_L = read_file(file_name,encode)
    word_freq = count_frequency(L)
    print("Successfully count word freqency!" + "(%s)" % file_name)
    
    word_seq, word_list = decide_seq_order(L)
    c_list = transfrom_wordlist_into_charlist(L)
    char_seq, char_list = decide_seq_order(c_list)
    char_freq = count_frequency(c_list)
    print("Successfully count char freqency!")
    
    pd_word= produce_data_frame(word_list, word_freq, word_seq,"word")
    another_word = pd_word.copy()
    pd_char= produce_data_frame(char_list, char_freq, char_seq,"char")
    data_frame = produce_wordRank_charRank_frame(pd_word,pd_char,longest_L)
    print("Successfully build data frames!")
    
    return data_frame, pd_char, another_word, longest_L

def N_gram_info(file_name, N, encode = "UTF-8"):
    '''This is only used to analysis N-gram words.
        
    paras:
    --------
    file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.
        
    N: int
      "N"-gram. 
      For example : a string, ABCDEFG (In Chinese, you don't know what's the 'words' of a string)
      in 2-gram = [AB, CD, EF, G];
      in 3-gram = [ABC, DEF, G];
      in 4-gram = [ABCD, EFG]
      
      two word compose a txt 'ABCDE EFGHI' (This case happended in English corpus)
      in 2-gram = [AB, CD, E, EF, GH, I];
      in 3-gram = [ABC, DE, EFG, HI];
      in 4-gram = [ABCD, E, EFGH, I]
    
    encode : encoding of your txt
    
    
    return:
    --------
    data_frame: pd.dataframe
      a big data frame contain the information of words and its compositition
    pd_char: pd.dataframe
      a data frame contain the information of characters
    another_word: pd.dataframe
      a data frame contain the information of words
    longest_L: int
      the biggest length of single word.
    
    '''
    L, longest_L = read_Ngram_file(file_name, N, encode)
    word_freq = count_frequency(L)
    print("Successfully count word freqency!" + "(%s)" % file_name)
    
    word_seq, word_list = decide_seq_order(L)
    c_list = transfrom_wordlist_into_charlist(L)
    char_seq, char_list = decide_seq_order(c_list)
    char_freq = count_frequency(c_list)
    print("Successfully count char freqency!")
    
    pd_word= produce_data_frame(word_list, word_freq, word_seq,"word")
    another_word = pd_word.copy()
    pd_char= produce_data_frame(char_list, char_freq, char_seq,"char")
    data_frame = produce_wordRank_charRank_frame(pd_word,pd_char,longest_L)
    print("Successfully build data frames!")
    
    return data_frame, pd_char, another_word, longest_L


def geometric_sequence(word, char):
    '''give geometric sequence {Hn} and {Vn}
    
    paras:
    ---
    word, char: pandas.DataFrame
        the output of info    
    
    returns:
    ---
    H: ndarray
        the geometric sequence of horizontal lines
    V: ndarray
        the sequence of vertical lines
      
    '''
    
    V = [0 for i in range(len(set(word['wordFreq'])))]
    H = [0 for i in range(len(set(char['charFreq'])))]
    
    Vf = sorted(set(word['wordFreq']))
    Hf = sorted(set(char['charFreq']))
    
    SVT = 0
    SHT = 0
    
    for i in range(len(set(word['wordFreq']))):
        #ref: Count how many values in a list that satisfy certain condition
        SV = sum(1 for cf in word['wordFreq'] if cf == Vf[i])
        SVT = SVT + SV
        V[i] = len(word['wordFreq']) - SVT + 1
    V[:0] = (max(word['wordRank']),)
        
    for i in range(len(set(char['charFreq']))):
        SH = sum(1 for wf in char['charFreq'] if wf == Hf[i])
        SHT = SHT + SH
        H[i] = len(char['charFreq']) - SHT + 1
    H[:0] = (max(char['charRank']),)
    
    return V, H
    

def draw_RRD_plot(big, word, char, longest, name, V, H, need_line = 'Y', number_of_lines = 4, Color = '#ff0000', SP = 'T', FORMAT = 'png'):
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
    big, word, char, longest: pandas.DataFrame
        the output of the function info()
    H, V: ndarray
        the output of the function check_const_ratio
           
    output:
        show a RRD plot
    
    '''
     
    fig, ax = plt.subplots()   
    if need_line == 'Y':
                
        Slice_number = 50 #this value decide the number of points on horizontal and vertical lines
        x_range = np.linspace(0, len(word), Slice_number)
        y_range = np.linspace(0, len(char), Slice_number)
        
        
        for i in range(number_of_lines):
            x_const = [V[i] for j in range(Slice_number)]#x_const =[V[i], V[i], ..., V[i]], Slice_number elements
            y_const = [H[i] for j in range(Slice_number)] #y_const =[H[i], H[i], ..., H[i]], Slice_number elements
            plt.plot(x_range, y_const) #plot y=H[i] on RRD plot
            plt.plot(x_const, y_range) #plot x=V[i] on RRD plot   
    
    color_list = ['#ff0000', '#CD00FF', '#ff00AB', '#ff004D', '#ff00F7', '#9100FF', '#4D00FF', '#0000FF', '#0066FF', '#00CDFF','#00FFCD', '#00FF5E','#80FF00','#EFFF00', '#FFB300']
    for i in range(longest): #draw 0th_char ~ longest_th_char on RRD plot
        str_position = [i + 1 for i in range(len(big[str(i) + "th_char_rank"]))] #position starting form 1 not 0
        plt.plot(str_position, big[str(i) + "th_char_rank"], 'o', markersize=3, color = Color, alpha = 0.7)
    plt.xlabel('word', size = 15)
    plt.ylabel('character', size = 15)
    
    plt.xlim([0, max(word['wordRank'])*11/10])
    plt.ylim([0, max(char['charRank'])*17/15])
    plt.title(name, size = 20)
    if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):
        fig.savefig('RRD of ' + name + '.' + FORMAT, dpi = 1000, format = FORMAT) #adjust dpi if you want the figure more clear
    plt.show()
    
    


def draw_density_plot(data, slice_number = 20, longest = 1):
    """input a pandas data frame, draw a density diagram of feature column, slice
    the diagram into slice_number equal pieces.
    
    Bugs: NaN can't be plot in hist2d, so only when longest = 1 then this function works
    """
    xx = []
    yy = []
    
    for i in range(longest):
        xx.extend([i + 1 for i in range(len(data[str(i) + "th_char_rank"]))]) #position starting form 1 not 0
        yy.extend([i  for i in data[str(i) + "th_char_rank"]])

    plt.hist2d(yy,xx, slice_number, cmap = plt.cm.jet)
    plt.colorbar()
    
    
def write_to_excel(big, word, char, name):
    """Write pandas dataFrame big, word, char to an excel file with the given filename
    """
    writer = pd.ExcelWriter(name + '.xlsx')
    big.to_excel(writer,'RRD')
    word.to_excel(writer,'word')
    char.to_excel(writer,'char')
    writer.save()


def read_file_generate_fake_constraint(constraint = 5, char_num = 2, out_file =  'fake1.txt', sample_word_num = 8000,
                            num_word_in_fake_scrip = 15000, 
                            alpha = 1.00001, noun = False):
    """Read "roc2.txt" file, and then generate a fake script satisfying Zipfs' law. All the words in 
    the output script share the same lenth char_num
    """
    CONSTRAINT = constraint
    SAMPLE_WORD_NUM = sample_word_num
    ALPHA = alpha
    NUM_WORD_IN_NOV = num_word_in_fake_scrip
    OUTPUT_FILE_NAME = out_file
    NOUN = noun
    CHAR_NUM = char_num
    
    zipf_gen =  ZipfGenerator(SAMPLE_WORD_NUM,ALPHA)
    f =  open("roc2.txt","r")

    world_list = []
    
    for line in f:
        line_split = line.split("\t")
        if NOUN:
            if 'N' in line_split[4]:
                world_list.append(line_split[3])
        else:
            #if len(line_split[3]) == CHAR_NUM:
                world_list.append(line_split[3])

    f.close()
    
    for item in world_list:
        if item == " ":
            world_list.remove(item)
    #######################################
    ##########produce fake words###########
    
    tmp_list = []
    for item in world_list:
        for e in list(item):
            if e not in tmp_list:
                tmp_list.append(e)
    char_count_dic = {}
    for c in tmp_list:
        char_count_dic[c] = 0
    

        
    
    list_2 = []
    tmp = ''
    for i in range(SAMPLE_WORD_NUM):
        for j in range(char_num):
            c = random.choice(tmp_list)
            char_count_dic[c] += 1
            if char_count_dic[c] >= CONSTRAINT:
                tmp_list.remove(c)
            tmp = tmp + c
        list_2.append(tmp)
        tmp = ''
    
    world_list = list_2[:]

    print("Words in corpus: " ,len(world_list))
    
    
    #######################################


    print("A corpus is successfully loaded.")
    
    random.shuffle(world_list)
    small_world_list = world_list[:]
    target_string_list = []

    for i in range(NUM_WORD_IN_NOV):
        num = zipf_gen.next()
        w = small_world_list[num]
        target_string_list.append(w+" ")
        
    f2 = open(OUTPUT_FILE_NAME , 'w')

    word_count = 0
    for item in target_string_list:
        if word_count < 20:
            f2.write(item)
            word_count += 1
        else:
            word_count = 0
            f2.write(item+"\n")
    f2.close()
    print("A fake script is successfully created !")
    print("--------------------")
    return None   


def read_file_generate_fake(char_num = 2, out_file =  'fake1.txt', sample_word_num = 8000,
                            num_word_in_fake_scrip = 15000, 
                            alpha = 1.00001, noun = False):
    """Read "roc2.txt" file, and then generate a fake script satisfying Zipfs' law. All the words in 
    the output script share the same lenth char_num
    """
    SAMPLE_WORD_NUM = sample_word_num
    ALPHA = alpha
    NUM_WORD_IN_NOV = num_word_in_fake_scrip
    OUTPUT_FILE_NAME = out_file
    NOUN = noun
    CHAR_NUM = char_num
    
    zipf_gen =  ZipfGenerator(SAMPLE_WORD_NUM,ALPHA)
    f =  open("roc2.txt","r")

    world_list = []
    
    for line in f:
        line_split = line.split("\t")
        if NOUN:
            if 'N' in line_split[4]:
                world_list.append(line_split[3])
        else:
            #if len(line_split[3]) == CHAR_NUM:
                world_list.append(line_split[3])

    f.close()
    
    for item in world_list:
        if item == " ":
            world_list.remove(item)
    #######################################
    ###these codes are optional 
    
    tmp_list = []
    for item in world_list:
        for e in list(item):
            tmp_list.append(e)
    random.shuffle(tmp_list)
    list_2 = []
    tmp = ''
    for e in tmp_list:
        tmp = tmp + e
        if len(tmp) == char_num:
            list_2.append(tmp)
            tmp = ''
    
    world_list = list_2

    print("words in a corpus: " ,len(world_list))
    
    
    #######################################


    print("A corpus is successfully loaded.")
    
    random.shuffle(world_list)
    small_world_list = world_list[-SAMPLE_WORD_NUM:]
    target_string_list = []

    for i in range(NUM_WORD_IN_NOV):
        num = zipf_gen.next()
        w = small_world_list[num]
        target_string_list.append(w+" ")
        
    f2 = open(OUTPUT_FILE_NAME , 'w')

    word_count = 0
    for item in target_string_list:
        if word_count < 20:
            f2.write(item)
            word_count += 1
        else:
            word_count = 0
            f2.write(item+"\n")
    f2.close()
    print("A fake script is successfully created !")
    print("--------------------")
    return None