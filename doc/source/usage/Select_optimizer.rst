5. Select Optimizer
-------------------

Choose the optimizer and set the parameters for the optimization process. AMOFMS provides various optimization algorithms to help automate this process, includeing :ref:`ParticleSwarmOptimizer`, :ref:`BayesianOptimizer`,  :ref:`GeneticOptimizer`, :ref:`SimplexOptimizer`. For example, use the ParticleSwarmOptimizer:

.. code-block:: python

    from AMOFMS.Optimization import ParticleSwarmOptimizer, Particle
    # num_particles do not set 1
    optimizer = ParticleSwarmOptimizer(objective_function=opt_loss_function, update_boundary_frequency=max_iter, bounds=opt_para_boundary, num_particles=num_particles, max_iter=max_iter, max_no_improvement_iters=max_iter)


