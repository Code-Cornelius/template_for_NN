# Ideas for continuing the project:

# Niels
* think of new tasks for Bianca

# Bianca


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