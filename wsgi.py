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
    def getArray(self):
      return self.predictor_array
    def getPredictor(self, n):
      return self.predictor_array[n]
    def deallocate(self, n):
      del self.predictor_array[n]

s = ""

class PredictorManager(BaseManager):
    pass


pmanager = None
#predictorAllocator = PredictorAllocator(0,100)




def applicatio(predictorAllocator, environ, start_response):
    global s
    ctype = 'text/plain'
    s = ""
    s += str(predictorAllocator)
    if environ['PATH_INFO'] == '/tests':
        s += predict.run_all_tests()
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        ctype = 'text/html'
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/predict_list':
        if predictorAllocator != None:
          s += str(os.getpid())
          s += str(predictorAllocator.getArray())
          s += str(predictorAllocator)
    if environ['PATH_INFO'] == '/predict_create':
        if predictorAllocator != None:
          s += str(os.getpid())
          s1 = environ['QUERY_STRING']
          s1 = s1.replace("%20"," ")
          d = parse_qs(s1)
#        s += str(d)
#        s += d["W"][0]
          Wout = eval(d["W"][0])
          step = eval(d["step"][0])
          n = predictorAllocator.allocate(int(d["points_per_network"][0]), Wout, int(d["num_layers"][0]), step, int(d["max_iterations"][0]))
          s+=" Predictor created "+ str(n) + str(predictorAllocator.getArray())
          ctype = 'text/html'
          s = s.replace("\n"," <br> ")
          s = s.replace("\r"," <br> ")
          response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/predict_remove':
        s1 = environ['QUERY_STRING']
        s1 = s1.replace("%20"," ")
        d = parse_qs(s1)
        predictorAllocator.deallocate(int(d["n"][0]))
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


from time import sleep




class MyAppClass:
    def __init__(self):
      self.predictorAllocator = None
    def __call__(self, environ, start_response):
      print "me: ", self, self.predictorAllocator
      return applicatio(self.predictorAllocator, environ, start_response)

PredictorManager.register('PManager', PredictorAllocator)
pmanager = PredictorManager()
pmanager.start()
application = MyAppClass()
predictorAllocator = pmanager.PManager()
application.predictorAllocator = predictorAllocator
f = open(os.environ['OPENSHIFT_DATA_DIR']+"myfile",'w')
f.write(str(os.getpid())) # python will convert \n to os.linesep
f.close() # you can omit in most cases as the de

#
# Below for testing only
#
if __name__ == '__main__':
    global predictorAllocator
    global pmanager
    global application
    from wsgiref.simple_server import make_server
    print "aaaaaa"
    httpd = make_server('localhost', 8051, application, handler_class = MyHandler)
    print "app: ", application.predictorAllocator
    print "app: ", application
    # Wait for a single request, serve it and quit.
    httpd.serve_forever()