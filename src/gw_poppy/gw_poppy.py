from poppy import Poppy
from poppy_bilby.utils import (
    get_inputs_from_bilby_pipe_ini,
    samples_from_bilby_result,
)

class GWPoppy(Poppy):

    @classmethod
    def from_bilby_pipe_ini(
        cls,
        *,
        ini_file,
        data_dump_file,
        suppress_bilby_logger: bool = True, 
        **kwargs
    ):
        inputs = get_inputs_from_bilby_pipe_ini(
            ini_file,
            data_dump_file,
            suppress_bilby_logger=suppress_bilby_logger,
        )
        return cls(
           log_likelihood=inputs.log_likelihood,
           log_prior=inputs.log_prior,
           dims=inputs.dims,
           parameters=inputs.parameters,
           prior_bounds=inputs.prior_bounds,
           periodic_parameters=inputs.periodic_parameters,
           **kwargs
        )
 
    def fit_from_bilby_result(self, result = None, result_file: str = None, **kwargs):
        from bilby.core.result import read_in_result
        if result is None:
            result = read_in_result(result_file)
        samples = samples_from_bilby_result(result, self.parameters)
        return self.fit(samples, **kwargs)
