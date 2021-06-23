# Ideas for continuing the project:

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
      

  
estim hyper param, see how it works, plots...


* Evolution plot there is a todo. in draw.
  
  * plot evol hawkes mseerrors todo for unique MSE
    
* evolution plot estimator _plot_finalisation
* Name of column in evolution plot drawing.
  
* for hawkes estim, not mandatory to have true values. in init


* take care of dependencies BETWEEN libraries.
  Ideally, create different folders ? 
  one big folder with the library
  and then put things together perhaps.