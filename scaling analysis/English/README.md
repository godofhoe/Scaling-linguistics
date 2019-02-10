A statistical tool to analyze the "scaling property" in English corpora
---------
## Fitting_MLE
A teaching material of maximum likelihood estimation (MLE).

## Run_case_by_case
Apply scaling analysis case by case. View this file for its details.

## Run_All
Apply scaling analysis on the txt files in data/Text

Tutorial
------
This tutorial will teach you how to use Run_All. Before you use Run_All, we suggest you try Run_case_by_case to know the functionalities of each function.

1. Put your segmented txt files in data/Text

2. In main(), there are several settings. You can see Run_case_by_case for their details.
* If you don't want to run certain functions, you can # them. For example, 
<pre><code>
  #FRD_plot(...)
</code></pre>
  
* Special cases: calculate SP and fitting scaling lines. You need to # three functions.
<pre><code>
  #f, flu = Plot_f(...)
  #Rf = rf(...)
  #fit_with_cut(...)
</code></pre>

3. In jupyter notebook, use Cell > Run all to excute the program.

4. If there is any bug, it will show in the bottom of this program. Please report us if you can not solve the bug.
