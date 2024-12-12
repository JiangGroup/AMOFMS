:class:`LossRecorder`
======================
A class for recording and retrieving losses for various terms across iterations during a training process.

Attributes
----------
- ``iteration_losses`` (dict): Stores the total and individual term losses for each iteration.
- ``iteration_opt_parameters`` (dict): Keeps track of the optimization parameters used in each iteration.
- ``header_written`` (bool): Indicates if the header has been written to the output file.

Methods
-------
- ``__init__(self)``: Initializes the LossRecorder class with necessary attributes.
- ``record_loss(self, iteration, opt_parameters, total_loss, each_term_loss, write_iteration_loss_filepath=None, note=None)``: Records the losses for a given iteration and optionally writes them to a specified file.
- ``get_iteration_loss(self, iteration)``: Retrieves the loss record for a specific iteration.
- ``get_losses(self)``: Returns a dictionary of all recorded losses for every iteration.
- ``get_term_loss(self, term)``: Retrieves a list of losses for a specified term across all recorded iterations.
- ``get_min_loss_and_iteration(self, term)``: Identifies the minimum loss value for a specified term across all iterations and the corresponding iteration number.
- ``write_losses_to_file(self, filepath)``: Writes the recorded losses to a file, formatting the output for readability.
