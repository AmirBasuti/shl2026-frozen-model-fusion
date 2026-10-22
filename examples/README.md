# Synthetic arithmetic example

`synthetic_probabilities.json` contains two invented rows. They are not outputs
of NuTime, Mantis or TiRex and have no associated ground-truth labels.

The first row assigns separate one-hot predictions to Still, Walk and Run.
The fusion therefore returns `[0.54, 0.29, 0.17, 0, 0, 0, 0, 0]` and Still.
The second similarly returns Bike. These are demonstrations of arithmetic,
not activity-recognition measurements or benchmark results.
