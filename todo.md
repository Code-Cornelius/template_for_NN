# Ideas for continuing the project:

# Todos:

* **APLOT**. The 3D plot have a different logic, due to the fact that axis is 3D axis. How to deal with it ? 3DAPlot?

  perhaps an idea, split APlot into: displayable plot -> grid_plot -> APlot.

  the grid_plot would take care of all the parameters, and in particular, would handle the dict_plot_param, dict_ax...
  the issue with some parameters is that for dict_plot_param they are just given to the plotter. It should be filtered
  first. This is the reason to problems in 3D axis.

  For example, the cmap is a pain. grid_plot could create a function that does this, everything a function needs to
  check before starting the inside:
  then one just needs to wrap all the function inside APlot. that is actually my main struggle: how to deal with
  function that can take any keyword argument and has a behavior depending on that? like colors, linewidth etc...

* **ESTIM HYPER PARAM**, see how it works, plots... essentially there is no way for now to test. So, the best thing
  would be:
    1. simulate a simple model (like sinus) on euler. Get 100 different config's performance. Then, that s where this
       estim kicks in.
    2. How do you go from the simulations / performance (in other words history estim) to HP estim?
    3. Efficient way to compare models, plotting etc. This will potentially lead to a new plotter: the scatter plot.

* **LIBRARIES**
    * take care of dependencies BETWEEN libraries. Ideally, create different folders ? one big folder with the library
      and then put things together perhaps.

    * Merge the two NN_template and Lib. Merging once the nn template is satisfying:
      that implies the architectures are good (this is ok) but also having the hyper parameters estimator functional.

# List of tasks to tackle:

* plot relplot hawkes mseerrors todo for unique MSE ( Niels will do) in draw.

* for hawkes estim, not mandatory to have true values. in init


  
