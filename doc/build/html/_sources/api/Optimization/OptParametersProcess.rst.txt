.. _OptParametersProcess:

:class:`OptParametersProcess`
==============================
A class for managing the optimization of parameters within molecular dynamics simulations.

Attributes
----------
- ``system_topology`` (dict): Stores the system's molecular topology information.
- ``opt_term_parse`` (dict): Contains details on which parameters are to be optimized.
- ``equivalent_term_id_list`` (list): Maps parameters to their equivalent terms for optimization purposes.
- ``equivalent_term_boundary_value_dict`` (dict): Specifies boundary values for equivalent optimization terms.
- ``opt_parameter_boundary_array`` (numpy.ndarray): An array containing the lower and upper boundaries for each optimization parameter.

Methods
-------
- ``__init__(self, system_topology, opt_term_parse, equivalent_term_id_list=None, equivalent_term_boundary_value_dict=None)``: Initializes the OptParametersProcess class with the system topology and optimization parameters.
- ``pack_opt_parameters_to_boundary_array(self)``: Converts the optimization parameters defined in the system topology into a structured array that delineates their lower and upper boundaries.
- ``unpack_updated_parameters_to_top(self, updated_parameters_array)``: Updates the system topology with the optimized parameter values obtained from the optimization process.
- ``adjust_for_equivalent_terms(self, boundary_array)``: (Private Method) Adjusts the parameter boundary array to account for equivalent terms as defined by `equivalent_term_id_list`.
- ``get_optimized_parameters(self)``: Retrieves the optimized parameters from the current system topology.
- ``save_optimized_parameters(self, filepath)``: Saves the optimized parameters to a specified file.
- ``load_optimized_parameters(self, filepath)``: Loads optimized parameters from a specified file into the system topology.



