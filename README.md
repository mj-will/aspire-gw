# poppy-gw

Gravitational-wave extensions to `poppy`.

## GWPoppy

`poppy-gw` provides `GWPoppy` a subclass of `Poppy` that has additional functionality
to enable training from, for example, `bibly` result files.

See the class for more details.

## Normalizing flows

`poppy-gw` provides an interface to the [`gwflow` package](https://github.com/mj-will/gwflow)
which implements GW-specific normalizing flows.
These can be used with `poppy` by specifying `flow_backend='gwflow'`.

**Note:** this requires have `gwflow` installed.
