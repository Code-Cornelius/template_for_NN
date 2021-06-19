# Ideas for continuing the project:

# Niels

to check correctness : 

* adapt the functions for the Estim_history
    * look into the saving data method. (it lies in data_processing). I left it half undone (as I was using it) because
      with estimator it will be much easier. The important thing is saving the information for each estimator in 2
      places:
        * saving the DF in a classical way (call super). you did it differently, can you change estimator as well please
          ? use super?
          
# Bianca

      
* Gru and lsdm
* can you check what happens when batch size is bigger than actual size of data? 
  I think it is fine, that the metrics are not multiplied by something too big, 
  neither are empty rows created but well...


* plot estimator. But for that we need to look at estimator and plot estimator. We can do that together in June after the exam.
Essentially, we can make a very trivial example, like a time estimator (can you do it ?) 
  that takes column time and any given parameter. 
  Then, we can start trying to plot it with the plot estimators, that way we can see what kind of plot we need.
  OK i have that great idea, estimator -> estimator time -> estimator array operation
  columns are time -> time, size, array type !
  Then what we want is some files where operations are performed (like append, some computations etc...)
  and then we fill it with some measurements and boom. Now we want to plot.
  What do we want to plot ?
  histogram. CHECK
  evolution wrt to a parameter? CHECK
  some other stats ? like what is the evolution of variance wrt the length ? CHECK.


# Todos:
* aplot. The 3D plot have a different logic, due to the fact that axis are 3D axis.
  How to deal with it ? 3DAPlot?
  
    perhaps an idea, split APlot into: displayable plot -> grid_plot -> APlot. 
  
      the grid_plot would take care of all the parameters, 
      and in particular, would handle the dict_plot_param, dict_ax...
      the issue with some parameters is that for dict_plot_param they are just given to the plotter. 
      It should be filtered first. This is the reason to problems in 3D axis. 
  For example, the cmap is a pain.
      grid_plot could create a function that does this, everything a function needs to check before starting the inside: 
      then one just needs to wraps all the function inside APlot.
      

  
* running some examples for checking that the functions are okay