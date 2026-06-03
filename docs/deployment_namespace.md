# Deployment Boundary

The public Open-WAM snapshot contains the research package named `open_wam`.

Robot deployment workspaces, live-rig scripts, camera synchronization probes,
and operator logs are not included in this first public release. Those
components depend on site-specific hardware, local credentials, and internal
runbooks, so they should stay outside the public package until they are
converted into stable, documented interfaces.

## Rule

- Use `open_wam` for research runtime, training, evaluation, policy variants,
  visual towers, and dataset adapters.
- Do not import deployment-only modules from package runtime code under
  `src/open_wam`.
- Add deployment code to the public repo only after it has a documented setup
  path, public-safe placeholders, and CI coverage that does not require private
  hardware.

## Migration Direction

Future cleanup can publish deployment support as an optional package or extra.
Until then, public docs should describe simulator and robot requirements at the
interface level rather than linking to internal deployment scripts.
