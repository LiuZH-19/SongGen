import os



def disable_proxy():
    # pass
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)



def enable_proxy():
    # pass
    os.environ['http_proxy']='XXX' 
    os.environ['https_proxy']='XXX' 
    os.environ['HTTP_PROXY']='XXX'
    os.environ['HTTPS_PROXY']='XXX'

