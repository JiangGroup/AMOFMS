4. Define Objective Function
----------------------------

Define the objective function for optimization. The Bottom-Up approach currently provides average potential force (Boltzmann inversion of radial distribution function g(r)) and force matching error calculations in :ref:`BottomUpObjectiveFunction`. The Top-Down approach provides density calculations, among others, available in the **tools.properties** module. Users can also define custom objective functions and calculation methods, including weights for each component of the total loss function.

Example of Objective Function (Loss Function) for :ref:`ParticleSwarmOptimizer`:

.. code-block:: python

    def opt_loss_function(particle: Particle):
        """
        This function computes the loss for a given particle in the optimization process using Particle Swarm Optimization (PSO).
        The loss is calculated based on Boltzmann inversion of potential energy and membrane properties (area per lipid, membrane thickness, and bending modulus).

        Parameters:
        - particle (Particle): An object representing a particle in the PSO process. `particle.position` is an array representing a set of parameters for the current iteration.

        Returns:
        - total_loss (float): The total computed loss for the given particle.
        - each_term_loss (dict): A dictionary containing the loss for each individual term (Ur, apl, thickness, and ka).
        """
        print(f'\n{particle.iter}-iter:  {particle.idx}-th particle processing...')
        
        # Unpack the particle's position (parameters) to update the system topology
        new_topology = opt_para.unpack_updated_parameters_to_top(updated_parameters_array=particle.position)
        
        # Define folder paths for storing iteration results
        iter_folder = os.path.join(opt_folder, f'iters/iter_{particle.iter}')
        idx_folder = os.path.join(iter_folder, f'{particle.idx}')

        # Update the system topology and optimization folder for the bottom-up object
        bottom_up_obj.update_system_topology(new_system_top=new_topology)
        bottom_up_obj.update_opt_folder(new_opt_folder=idx_folder)

        total_loss = 0
        each_term_loss = {}

        # Compute the Boltzmann inversion loss
        print('\nComputing Ur Boltzmann inversion loss...')
        bottom_up_obj.run_cg_simulation(
            initial_gro=initial_gro,
            fg_resname_list=resname_cg_from_fg_coord,
            mdp_folder=mdp_folder,
            index_file=index_file,
            table_file=None,
            cg_simulation_folder=idx_folder,
            em=run_em,
            em_double_version=em_double,
            anneal=run_anneal,
            gpu_acceleration=False,
            eq=run_eq,
            prod=run_prod,
            nt=cpu_nt,
            gpu_id=None
        )
        
        # Calculate the coarse-grained pair potentials using Boltzmann inversion
        cg_pair_Ur_list = bottom_up_obj.Boltzmann_inversion(
            rdf_pairs_list=cg_rdf_pairs_list,
            tag='cg',
            Temperature=temperature,
            bin_width=rdf_binwidth,
            max_distance=rdf_cutoff
        )
        
        # Calculate the loss based on the difference between fine-grained and coarse-grained pair potentials
        Ur_loss = Ur_loss_ratio * square_diff_of_elements_in_2D_list(list1=fg_pair_Ur_list, list2=cg_pair_Ur_list)
        each_term_loss.update({'Ur': Ur_loss})
        total_loss += Ur_loss

        # Compute the membrane properties loss
        print('\nComputing membrane properties...')
        mem = MembraneProperties(
            topology=bottom_up_obj.cg_topology,
            trajectory=bottom_up_obj.cg_trajectory
        )
        
        # Calculate area per lipid, membrane thickness, and bending modulus
        cg_apl, _ = mem.compute_apl(headgroup_selection=head_group_expression)
        cg_thickness, _ = mem.compute_membrane_thickness(headgroup_selection=head_group_expression)
        cg_ka = mem.compute_Ka(head_group_expression, temperature)

        print(f'{particle.iter}-iter {particle.idx}-th particle APL (Angstrom**-2): \nexp: {exp_apl} cg: {cg_apl}')
        print(f'{particle.iter}-iter {particle.idx}-th particle Thickness (Angstrom): \nexp: {exp_thickness} cg: {cg_thickness}')
        print(f'{particle.iter}-iter {particle.idx}-th particle Ka (mN/m): \nexp: {exp_ka} cg: {cg_ka}')

        # Calculate the loss for each membrane property
        apl_loss = apl_loss_ratio * abs(exp_apl - cg_apl)  # https://doi.org/10.1021/acs.jpcb.6b01870
        thickness_loss = thickness_loss_ratio * abs(exp_thickness - cg_thickness)  # https://doi.org/10.1063/1.4936909
        ka_loss = ka_loss_ratio * abs(exp_ka - cg_ka)

        each_term_loss.update({'apl': apl_loss})
        each_term_loss.update({'thickness': thickness_loss})
        each_term_loss.update({'ka': ka_loss})
        total_loss = total_loss + apl_loss + thickness_loss + ka_loss

        # Write the results to a temporary file
        with open(tmp_result, 'a+') as f:
            line = f'{particle.iter:<12}  {particle.idx:<12}  {Ur_loss:<12}  {cg_apl:<12}  {cg_thickness:<12} {cg_ka:<12}\n'
            f.write(line)

        print(f'\n{particle.iter}-iter:  {particle.idx}-th particle Done!')
        return total_loss, each_term_loss

.. note::

    **Boltzmann Inversion**: The Boltzmann inversion method is used to derive a potential of mean force (PMF) from the radial distribution function (RDF) \( g(r) \). The relationship is given by:

    .. math::
        \Delta U(r) = \left\| U(r)^{\text{CG}}  - U(r)^{\text{AA}}\right\|;\quad
        U(r) = -k_B T \ln g(r)

    where:
    
    .. math::
      U(r) \quad &\text{is the potential of mean force at distance} \ r;\\
      k_B \quad &\text{is the Boltzmann constant}; \\
      T \quad &\text{is the temperature}; \\
      g(r) \quad &\text{is the radial distribution function}.

    **Force Matching**: The force matching method involves comparing the forces from a coarse-grained (CG) model to those from an atomistic model. The force matching error is calculated as:

    .. math::
      \Delta F = \sum_{i=1}^{N} \left\| \mathbf{F}_i^{\text{CG}} - \mathbf{F}_i^{\text{AA}} \right\|

    where:

   .. math::
      \Delta F \quad &\text{is the force matching error}; \\
      \mathbf{F}_i^{\text{CG}} \quad &\text{is the force on CG group} \quad i \quad \text{in the CG model}; \\
      \mathbf{F}_i^{\text{AA}} \quad &\text{is the force on CG group} \quad i \quad \text{in the atomistic model}; \\
      N \quad &\text{is the number of particles}.


