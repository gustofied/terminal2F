from .payments import tools as payment_tools, names_to_functions as payment_functions
from .code_tools import tools as code_tools, names_to_functions as code_functions

tools = [*payment_tools, *code_tools]
names_to_functions = {**payment_functions, **code_functions}
