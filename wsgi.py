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
#
# IMPORTANT: Put any additional includes below this line.  If placed above this
# line, it's possible required libraries won't be in your searchable path
#

class PredictorAllocator:
    def __init__(self, n1, n2):
      self.predictor_array = { }
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

predictorAllocator = PredictorAllocator(0,100)

def application(environ, start_response):
    global predictorAllocator
    global s
    glo
    ctype = 'text/plain'
    if environ['PATH_INFO'] == '/tests':
        s = predict.run_all_tests()
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        ctype = 'text/html'
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/predict_create':
        s += environ['QUERY_STRING']
        s = s.replace("%20"," ")
        d = parse_qs(s)
        s += str(d)
        s += d["W"][0]
        Wout = eval(d["W"][0])
        step = eval(d["step"][0])
#        n = predictorAllocator.allocate(int(d["points_per_network"][0]), Wout, int(d["num_layers"][0]), step, int(d["max_iterations"][0]))
        s+=" Predictor created "+ str(n)
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

#
# Below for testing only
#
if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    httpd = make_server('localhost', 8051, application)
    # Wait for a single request, serve it and quit.
    httpd.handle_request()
