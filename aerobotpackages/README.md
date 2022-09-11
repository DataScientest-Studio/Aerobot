# Working with project-specific packages

by Ioannis Stasinopoulos, 11.09.2022

Define as many modules as you need, like in 'aerobotfunctions.py', which is a module containing function definitions.

Then, in the '__init__.py' file, add:
```
from [your module name].py include *
```
