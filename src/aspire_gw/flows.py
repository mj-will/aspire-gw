import logging
from dataclasses import dataclass, field

from aspire.flows.torch.flows import BaseTorchFlow, ZukoFlow
from aspire.history import FlowHistory

logger = logging.getLogger(__name__)


@dataclass
class GWFlowHistory(FlowHistory):
    """History class for GWFlow.

    Includes pre-training history if applicable.
    """

    pre_training_loss: list[float] = field(default_factory=list)
    pre_training_val_loss: list[float] = field(default_factory=list)

    def plot_loss(self) -> None:
        """Plot the training and validation loss, including pre-training if available."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        loss = self.training_loss
        val_loss = self.validation_loss
        if self.pre_training_loss:
            loss = self.pre_training_loss + self.training_loss
            val_loss = self.pre_training_val_loss + self.validation_loss
            plt.axvline(
                len(self.pre_training_loss),
                color="k",
                ls="--",
                label="End of pre-training",
            )
        plt.plot(loss, label="Training loss")
        plt.plot(val_loss, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        return fig


class GWFlow(ZukoFlow):
    """Wrapper gwflow to be used with aspire.

    Can be used by specifying `flow_backend='gwflow'` in `Aspire`.

    Parameters
    ----------
    dims : int
        Dimensionality of the data.
    data_transform : aspire.transforms.Transform, optional
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

        logger.info(f"Initializing GWCalFlow with kwargs: {kwargs}")
        self.flow = GWCalFlow(parameters=parameters, **kwargs)
        logger.info(f"Initialized GWCalFlow: \n {self.flow}\n")

    def fit(self, x, pre_train_cal: int = 0.0, **kwargs):
        """Fit the flow to the data.

        Parameters
        ----------
        pretrain_cal : int, optional
            If > 0, run this many epochs training only the calibration network
            before fitting the full joint flow.
        *args, **kwargs :
            Passed to the `fit` method of the underlying flow.
        """
        if pre_train_cal:
            logger.info(
                f"Pretraining calibration network for {pre_train_cal} epochs."
            )

            # Freeze GW flow
            for p in self.flow.flow_gw.parameters():
                p.requires_grad = False

            # Run the normal fit loop for calibration only
            pre_train_kwargs = kwargs.copy()
            pre_train_kwargs["n_epochs"] = pre_train_cal
            pre_train_history = super().fit(x, **pre_train_kwargs)

            # Unfreeze GW flow
            for p in self.flow.flow_gw.parameters():
                p.requires_grad = True
        else:
            pre_train_history = FlowHistory()

        # Now run the full joint fit
        flow_history = super().fit(x, **kwargs)

        history = GWFlowHistory(
            training_loss=flow_history.training_loss,
            validation_loss=flow_history.validation_loss,
            pre_training_loss=pre_train_history.training_loss
            if pre_train_cal
            else [],
            pre_training_val_loss=pre_train_history.validation_loss
            if pre_train_cal
            else [],
        )
        return history
