import logging

from poppy.flows.torch.flows import BaseTorchFlow, ZukoFlow

logger = logging.getLogger(__name__)


class GWFlow(ZukoFlow):
    """Wrapper gwflow to be used with poppy."""

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
