3. Identify Parameters for Optimization
---------------------------------------

Identify the parameters and their ranges to optimize (e.g., LJ parameters, equilibrium bond lengths, bond angles, dihedral angles, corresponding force constants, charges, and masses). Determine which parameters are equivalent during the optimization process. 0 repersents this parameter will not be optimized.

The `opt_term_parse` dictionary specifies the parameters and their optimization settings for the optimization process. Below is a breakdown of its structure:

- **molecules** (list of dicts): Each dictionary represents a molecule type with its parameters to be optimized.
  
  - **mol_name** (str): The name of the molecule.
  - **bond_parameters** (dict): Parameters for optimizing bond lengths between atoms. Each key represents a bonded atom pair, and the value is a list containing the optimization settings for bond length and force constant.
  - **angle_parameters** (dict): Parameters for optimizing bond angles between atoms. Each key represents an atom triplet defining the angle, and the value is a list containing the optimization settings for angle value and force constant.

- **lj_cross_terms** (dict): Parameters for optimizing Lennard-Jones interaction terms between atom types. Each key represents a tuple of atom types, and the value is a list containing the optimization settings for epsilon and sigma.

Here is an example of the `opt_term_parse` dictionary:

.. code-block:: python

    opt_term_parse = {
        'molecules': [
            {
                'mol_name': 'POPC',
                'bond_parameters': {(0, 1): [0, 0.1], (1, 2): [0, 0.1],
                                    ...,
                                    (10, 11): [0, 0.1]},
                'angle_parameters': {(4, 5, 8): [0, 0.1],
                                     ...,
                                     (0, 1, 2): [0, 0.1]}
            }
        ],
        'lj_cross_terms': {('C3', 'NA'): [0.1, 0.1], ('C3', 'QA'): [0.1, 0.1],
                           ...,
                           ('Q0', 'Q0'): [0.1, 0.1], ('C1', 'C1'): [0.1, 0.1]}
    }

Here are some examples illustrating the optimization settings:

- **Bond Parameters Optimization**:

  - For the bond between atoms 0 and 1 in the molecule 'POPC', the bond length and force constant are optimized with a percentage change of 0.1 relative to their original values. For the bond between atoms 0 and 1 ((0, 1): [0, 0.1]) in the molecule 'POPC', the bond length and force constant are optimized with a percentage change of 10 % relative to their original values. The original values from `system_topology` are (0, 1): [0.4656, 1250.0]. Therefore, for the bond between atoms with IDs 0 and 1, the equilibrium length (0.4656 nm) will not be changed (not be optimized), and the corresponding force constant will be optimized in the range [1250 * (1 - 0.1), 1250 * (1 + 0.1)].

- **Angle Parameters Optimization**:

  - The angle between atoms 4, 5, and 8 in the molecule 'POPC' is optimized with a 10 % change in both angle value and force constant. Likewise, for the angle defined by atoms 4, 5, and 8 ((4, 5, 8): [0, 0.1]) in the molecule 'POPC', the angle and force constant are optimized with a percentage change of 0.1 relative to their original values. The original values from `system_topology` are (4, 5, 8): [105.2565, 25.0]. Therefore, for the angle defined by atoms with IDs 4, 5, and 8, the equilibrium angle (105.2565 degrees) will not be changed (not be optimized), and the corresponding force constant will be optimized in the range [25.0 * (1 - 0.1), 25.0 * (1 + 0.1)].

- **Lennard-Jones Cross Terms Optimization**:

  - For the LJ interaction between atom types 'Q0' and 'Q0' (('Q0', 'Q0'): [0.1, 0.1]), the LJ parameters (sigma and epsilon) are optimized with a percentage change of 0.1 relative to their original values. The original values from `system_topology` are ('Q0', 'Q0'): [0.1, 0.1]. Therefore, for the LJ interaction between atom types 'Q0' and 'Q0', both sigma and epsilon will be optimized in the range [0.1 * (1 - 0.1), 0.1 * (1 + 0.1)].
  
These examples demonstrate how the `opt_term_parse` dictionary defines the optimization settings for various parameters in the system topology. Then, the AMOFMS tool can parse the `opt_term_parse` dictionary using the :ref:`OptParametersProcess` module.

.. code-block:: python

   from AMOFMS.Optimization import OptParametersProcess

   # Define equivalent_term_id_list and equivalent_term_boundary_value_dict
   # equivalent_term_id_list is a list that maps optimization parameters to their equivalent terms,
   # allowing for grouped optimization of parameters that should be treated as equivalent.
   # For example, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates that parameters with IDs 0-10 are equivalent and will change simultaneously during optimization.
   equivalent_term_id_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1, 2, 1] + list(range(3, 59))
   equivalent_term_boundary_value_dict = {}

   # Create an instance of OptParametersProcess using system_topology, opt_term_parse,
   # equivalent_term_id_list, and equivalent_term_boundary_value_dict as parameters
   opt_para = OptParametersProcess(system_topology=system_topology, opt_term_parse=opt_term_parse,
                                   equivalent_term_id_list=equivalent_term_id_list,
                                   equivalent_term_boundary_value_dict=equivalent_term_boundary_value_dict)
   
   # Obtain opt_parameter_boundary_array and new_topology using the OptParametersProcess instance
   opt_para_boundary = opt_para.opt_parameter_boundary_array
   new_topology = opt_para.unpack_updated_parameters_to_top(updated_parameters_array=particle.position)                  

.. note::

   - The `equivalent_term_id_list` parameter is a list that maps optimization parameters to their equivalent terms, allowing for grouped optimization of parameters that should be treated as equivalent. In this case, parameters with the same ID will be optimized together, meaning they will change simultaneously during the optimization process.

   - The `equivalent_term_boundary_value_dict` parameter is a dictionary specifying boundary values for the equivalent terms defined in `equivalent_term_id_list`.
   
