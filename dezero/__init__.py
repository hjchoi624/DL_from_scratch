is_simple_core =True

if is_simple_core:
    from DL_from_scratch.dezero.core_simple import Variable
    from DL_from_scratch.dezero.core_simple import Function
    from DL_from_scratch.dezero.core_simple import using_config
    from DL_from_scratch.dezero.core_simple import no_grad
    from DL_from_scratch.dezero.core_simple import as_array
    from DL_from_scratch.dezero.core_simple import as_variable
    from DL_from_scratch.dezero.core_simple import setup_variable
else:
    from dezero.core import Variable
    from dezero.core import Function  



setup_variable()
