#!/usr/bin/python
import os
import predict
from random import randint
from predict import run_all_tests
from predict import Predictor
from cgi import parse_qs
virtenv = os.environ['OPENSHIFT_PYTHON_DIR'] + '/virtenv/'
virtualenv = os.path.join(virtenv, 'bin/activate_this.py')
try:
    execfile(virtualenv, dict(__file__=virtualenv))
except IOError:
    pass

from multiprocessing.managers import BaseManager
from multiprocessing import Process, Queue

#
# IMPORTANT: Put any additional includes below this line.  If placed above this
# line, it's possible required libraries won't be in your searchable path
#

class PredictorAllocator:
    def __init__(self, n1=0, n2=100):
      self.predictor_array = {}
      self.n1 = n1
      self.n2 = n2
    def allocate(self, points_per_network, W, num_layers, step, max_iterations):
      n = randint(self.n1,self.n2)
      while n in self.predictor_array:
        n = randint(self.n1,self.n2)
        self.predictor_array[n] = Predictor(points_per_network, W, num_layers, step, max_iterations)
      return n
    def getPredictor(self, n):
      return self.predictor_array[n]
    def deallocate(self, n):
      del self.predictor_array[n]

s = ""

class PredictorManager(BaseManager):
    pass

PredictorManager.register('PManager', PredictorAllocator)


#predictorAllocator = PredictorAllocator(0,100)

predictorAllocator = None

def application(environ, start_response):
    global predictorAllocator
    global s
    ctype = 'text/plain'
    PredictorManager.register('PManager', PredictorAllocator)
    pmanager = PredictorManager(address=('',50000), authkey='')
#    pmanager.connect()
#    predictorAllocator = pmanager.PManager()
    if environ['PATH_INFO'] == '/tests':
        s += predict.run_all_tests()
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        ctype = 'text/html'
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/predict_list':
        s += str(os.getpid())
        s += str(predictorAllocator.predictor_array)
        s += str(predictorAllocator)
    if environ['PATH_INFO'] == '/predict_create':
        s += str(os.getpid())
        s1 = environ['QUERY_STRING']
        s1 = s1.replace("%20"," ")
        d = parse_qs(s1)
#        s += str(d)
#        s += d["W"][0]
        Wout = eval(d["W"][0])
        step = eval(d["step"][0])
        n = predictorAllocator.allocate(int(d["points_per_network"][0]), Wout, int(d["num_layers"][0]), step, int(d["max_iterations"][0]))
        s+=" Predictor created "+ str(n) + str(predictorAllocator.predictor_array)
        ctype = 'text/html'
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/predict_remove':
        s1 = environ['QUERY_STRING']
        s1 = s1.replace("%20"," ")
        d = parse_qs(s1)
        predictor.Allocator.deallocate(int(d["n"][0]))
        s+=" Predictor removed "+ d["n"][0]
        ctype = 'text/html'
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/health':
        response_body = "1"
    elif environ['PATH_INFO'] == '/env':
        response_body = ['%s: %s' % (key, value)
                    for key, value in sorted(environ.items())]
        response_body = '\n'.join(response_body)
    else:
        ctype = 'text/html'
        response_body = '<html><body>' + s + '</body></html>'
    status = '200 OK'
    response_headers = [('Content-Type', ctype), ('Content-Length', str(len(response_body)))]
    #
    start_response(status, response_headers)
    return [response_body]

from wsgiref.handlers import SimpleHandler

class MyHandler(SimpleHandler):
  def __init__(self, stdin, stdout, stderr, environ, multithread=False, multiprocess=False):
    SimpleHandler.__init__(self, stdin, stdout, stderr, environ, multithread, multiprocess)

#
# Below for testing only
#
if __name__ == '__main__':
    global predictorAllocator
    from wsgiref.simple_server import make_server
    pmanager = PredictorManager(address=('',50000), authkey='')
    s = pmanager.get_server()
    p = Process(target=s.serve_forever, args=())
    p.start()
    httpd = make_server('localhost', 8051, application, handler_class = MyHandler)
    # Wait for a single request, serve it and quit.
    httpd.serve_forever()