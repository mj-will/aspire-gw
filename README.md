# aspire-gw

Gravitational-wave extensions to `aspire`.

## GWAspire

`aspire-gw` provides `GWAspire` a subclass of `Aspire` that has additional functionality
to enable training from, for example, `bibly` result files.

See the class for more details.

## Normalizing flows

`aspire-gw` provides an interface to the [`gwflow` package](https://github.com/mj-will/gwflow)
which implements GW-specific normalizing flows.
These can be used with `aspire` by specifying `flow_backend='gwflow'`.

**Note:** this requires have `gwflow` installed.
