import logging

from poppy.flows.torch.flows import BaseTorchFlow, ZukoFlow

logger = logging.getLogger(__name__)


class GWFlow(ZukoFlow):
    """Wrapper gwflow to be used with poppy.

    Can be used by specifying `flow_backend='gwflow'` in `Poppy`.

    Parameters
    ----------
    dims : int
        Dimensionality of the data.
    data_transform : poppy.transforms.Transform, optional
        Data transform to apply to the data before fitting the flow.
    seed : int, optional
        Random seed for reproducibility, by default 1234.
    device : str, optional
        Device to use for training, by default "cpu".
    parameters : list[str]
        List of parameter names corresponding to the dimensions. Must be
        provided if using GWCalFlow.
    """

    def __init__(
        self,
        dims,
        data_transform=None,
        seed=1234,
        device: str = "cpu",
        parameters: list[str] = None,
        **kwargs,
    ):
        from gwflow import GWCalFlow

        BaseTorchFlow.__init__(
            self,
            dims=dims,
            device=device,
            data_transform=data_transform,
            seed=seed,
        )

        if hidden_features := kwargs.pop("hidden_features", None):
            kwargs["hidden_features"] = list(map(int, hidden_features))
        self.flow = GWCalFlow(parameters=parameters, **kwargs)
        logger.info(f"Initialized GWCalFlow: \n {self.flow}\n")
