import matplotlib.pyplot as plt
import numpy as np
import pytest

from aspire_gw.flows import GWFlow, GWFlowHistory


def test_gwflow_initialization():
    flow = GWFlow(
        dims=4,
        parameters=["mass_1", "mass_2", "spin_1z", "recalib_amplitude_H1_0"],
        seed=42,
        device="cpu",
    )
    assert isinstance(flow, GWFlow)
    assert flow.dims == 4


@pytest.mark.parametrize("calibration_model", ["gaussian", "nn"])
@pytest.mark.parametrize("pre_train_cal", [None, 5])
def test_gwflow_fit(calibration_model, pre_train_cal):
    flow = GWFlow(
        dims=4,
        parameters=["mass_1", "mass_2", "spin_1z", "recalib_amplitude_H1_0"],
        seed=42,
        device="cpu",
        calibration_model=calibration_model,
    )
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 4))
    history = flow.fit(
        data,
        n_epochs=5,
        pre_train_cal=pre_train_cal,
        batch_size=10,
    )
    assert hasattr(flow, "flow")
    assert isinstance(history, GWFlowHistory)


def test_gwflow_history():
    history = GWFlowHistory(
        training_loss=[0.1, 0.2],
        validation_loss=[0.2, 0.3],
        pre_training_loss=[0.05, 0.1],
        pre_training_val_loss=[0.1, 0.15],
    )

    fig = history.plot_loss()
    assert fig is not None
    plt.close("all")
