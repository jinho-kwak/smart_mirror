# abort 임의 정의
from werkzeug.exceptions import HTTPException, default_exceptions, Aborter
from werkzeug.http import HTTP_STATUS_CODES

"""
- Fridge abort header sentence -
600 : LoadCell Error
601 : Product_count Error
602 : Inference Error
"""
HTTP_STATUS_CODES[600] = '(600)Interminds LoadCell Error'
HTTP_STATUS_CODES[601] = '(601)Interminds Product Error'
HTTP_STATUS_CODES[602] = '(602)Interminds Inference Error'

"""
- Cigar abort header sentence -
700 : Default Error
"""
HTTP_STATUS_CODES[700] = '(700)Interminds Default Error'

# Fridge abort 정의 class
class f_loadcell_err(HTTPException):
    code = 600
    description = 'loadcell_err'

class f_product_err(HTTPException):
    code = 601
    description = 'product_err'

class f_inference_err(HTTPException):
    code = 602
    description = 'inference_err'

# Fridge abort 정의 class
class c_default_err(HTTPException):
    code = 700
    description = 'default_err'

# Fridge abort Except define.
default_exceptions[600] = f_loadcell_err
default_exceptions[601] = f_product_err
default_exceptions[602] = f_inference_err

# Cigar abort Except define.
default_exceptions[700] = c_default_err

# abort 선언
abort = Aborter()