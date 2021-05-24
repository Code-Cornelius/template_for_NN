# Ideas for continuing the project:

# Niels
* New function for the project
* Estimator


# Bianca
* Estim_history
* adapt the functions for the Estim_history  
    * look into the saving data method. (it lies in data_processing).
      I left it half undone (as I was using it) because with estimator it will be much easier.
      The important thing is saving the information for each estimator in 2 places: 
            * saving the DF in a classical way (call super).
            * saving the fields as a list in a separate file.
      there is the method read_list_of_ints_from_path in data_processing that could do the trick?
* think about not predicting multiple times
By the way, storing the parameters (training ones) is not that important. 
  Focus on the rest first. 
  Saving the parameters will make the estim_hyper_param way easier though! 
  (but it is a side project).
  Indeed, then you can make a function that takes 
  a bunch of estim_history and create one estim_hyper_param.
  
* can you check what happens when batch size is bigger than actual size of data? 
  I think it is fine, that the metrics are not multiplied by something too big, 
  neither are empty rows created but well...


* Estim_hyper_param
* plot estimator. 


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
      
  GRU LSTM refactor ?
  
* running some examples for checking that the functions are okay