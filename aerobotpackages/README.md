# Working with project-specific packages

by Ioannis Stasinopoulos, 11.09.2022

Define as many modules as you need, like in 'aerobotfunctions.py', which is a module containing function definitions.

Then, in the '__init__.py' file, add:
```
from .[your module name] import *
```
This way, you do not need to explicitely import the new module in your python code. 
Simply importing aerobotpackages will include the new modules.
