# Working with project-specific packages

by Ioannis Stasinopoulos, 02.02.2023

Define as many modules as you need, like in 'streamlittools.py', which is a module containing function definitions that are necessary for our streamlit demo.
The custom-package should 'bring' along its necessary packages: make sure you import within the .py file all the packages that your functions / classes need, e.g. steamlit, os etc. 

Then, in the '__init__.py' file, add:
```
from .[your module name] import *
```
This way, you do not need to explicitely import the new module in your python code: importing ```streamlitpackages``` will automatically import all the modules.
