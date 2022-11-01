# Working with project-specific packages

by Ioannis Stasinopoulos, 11.09.2022

Define as many modules as you need, like in 'BERTTools.py', which is a module containing class and function definition for the transformer model.
The custom-package should 'bring' along its necessary packages: make sure you import within the .py file all the packages that your functions / classes need, e.g. numpy, pandas etc. 

Then, in the '__init__.py' file, add:
```
from .[your module name] import *
```
This way, you do not need to explicitely import the new module in your python code: importing ```aerobotpackages``` will automatically import all the modules.
