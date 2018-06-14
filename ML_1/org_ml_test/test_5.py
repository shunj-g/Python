from queue import Queue
from functools import wraps
def apply_async(func,args,callback):
    #conpute the result
    result = func(*args)
    #Invoke the callback with the result
    callback(result)

class Async:
    def __init__(self,func,args):
        self.func = func
        self.args = args

def inlined_async(func):
    @wraps(func)
    def wrapper(*args):
        f = func(*args)
        result_queue = Queue()
        result_queue.put(None)
        while True:
            result = result_queue.get()
            try:
                a = f.send(result)
                apply_async(a.func,a.args,callback=result_queue.put)
            except StopIteration:
                break
    return wrapper


#def print_result(result):
#    print('Got:',result)

def add(x,y):
    return x+y

@inlined_async
def test():
    r = yield Async(add,(2,3))#generator
    print(r)
    r = yield Async(add,('hello','world'))
    print(r)
    for n in range(10):
        r = yield Async(add,(n,n))
        print(r)
    print('Goodbye')
#apply_async(add,(2,3),callback=print_result)
#apply_async(add,('hello','world'),callback=print_result)
test()