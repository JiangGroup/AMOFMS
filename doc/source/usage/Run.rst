6. Run Optimization and Save Log
--------------------------------

Run the optimization process and save the log for analysis.

.. code-block:: python

    best_para, best_score, recorder = optimizer.optimize_mpi(max_processes=max_processes)  # parallelize
    # best_para, best_score, recorder = optimizer.optimize() # non-mpi
    recorder.write_losses_to_file(filepath=loss_dat)
    
