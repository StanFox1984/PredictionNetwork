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
import json
import pickle
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
    def load_from_file(self):
      if os.path.isfile(os.environ['OPENSHIFT_DATA_DIR']+"myfile"):
        f = open(os.environ['OPENSHIFT_DATA_DIR']+"myfile",'rb')
        self.predictor_array = pickle.load(f)
        f.close()
    def save_to_file(self):
      f = open(os.environ['OPENSHIFT_DATA_DIR']+"myfile",'wb')
      pickle.dump(self.predictor_array, f)
      f.close() # you can omit in most cases as the de
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
predictorAllocator = PredictorAllocator(0,100)

def handle_predict_list(environ, predictorAllocator):
    s = ""
    s1 = ""
    if predictorAllocator != None:
#        s1 += str(os.getpid())
      s1 += str(predictorAllocator.getArray())+"\n"
#        s1 += str(predictorAllocator)
      s1 = s1.replace("<", " ")
      s1 = s1.replace(">", " ")
      s+=s1
    response_body = '<html><body>' + s + '<script>\nalert("predict_list!!!")\n</script></body></html>'
    return response_body

def application(environ, start_response):
    global s
    ctype = 'text/html'
    s = ""
    s1 = ""
#    s += str(predictorAllocator)
    predictorAllocator.load_from_file()
    if environ['PATH_INFO'] == '/tests':
        s += predict.run_all_tests()
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        ctype = 'text/html'
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
    if environ['PATH_INFO'] == '/predict_list':
        response_body = handle_predict_list(environ, predictorAllocator)
    if environ['PATH_INFO'] == '/predict_create':
        if predictorAllocator != None:
#          s += str(os.getpid())
          s1 = environ['QUERY_STRING']
          s1 = s1.replace("%20"," ")
          d = parse_qs(s1)
#        s += str(d)
#        s += d["W"][0]
          Wout = eval(d["W"][0])
          step = eval(d["step"][0])
          n = predictorAllocator.allocate(int(d["points_per_network"][0]), Wout, int(d["num_layers"][0]), step, int(d["max_iterations"][0]))
          s+=" Predictor created "+ str(n)+"\n"
          ctype = 'text/html'
          s = s.replace("\n"," <br> ")
          s = s.replace("\r"," <br> ")
          response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/predict_study':
        if predictorAllocator != None:
#          s += str(os.getpid())
          s1 = environ['QUERY_STRING']
          s1 = s1.replace("%20"," ")
          d = parse_qs(s1)
#        s += str(d)
#        s += d["W"][0]
          n = eval(d["n"][0])
          X = eval(d["X"][0])
          Y = eval(d["Y"][0])
          p = predictorAllocator.getPredictor(n)
          if p != None:
            p.study(X,Y)
            s+=" Predictor study "+ str(n)+"\n"
          else:
            s+=" Not found" + str(n)+"\n"
          ctype = 'text/html'
          s = s.replace("\n"," <br> ")
          s = s.replace("\r"," <br> ")
          response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/predict':
        if predictorAllocator != None:
          #s += str(os.getpid())
          s1 = environ['QUERY_STRING']
          s1 = s1.replace("%20"," ")
          d = parse_qs(s1)
#        s += str(d)
#        s += d["W"][0]
          n = eval(d["n"][0])
          depth = eval(d["depth"][0])
          X = eval(d["X"][0])
          Yout = [ ]
          P = [ ]
          _classes = [ ]
          s1 = ""
          p = predictorAllocator.getPredictor(n)
          if p != None:
            p.predict_p_classes(X, Yout, P, depth, _classes)
            s1+="Predict "+ str(n) + ": " + str(X)+"\n"
            s1+="X:" + str(P) + "\n"
            s1+="Y:" + str(Yout) + "\n"
            for c in _classes:
              s1+="Class:" + str(c) + "\n"
          else:
            s1+=" Not found" + str(n)
          s1 = s1.replace("<", " ")
          s1 =s1.replace(">", " ")
          s+=s1
          ctype = 'text/html'
          s = s.replace("\n"," <br> ")
          s = s.replace("\r"," <br> ")
          response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/predict_remove':
        s1 = environ['QUERY_STRING']
        s1 = s1.replace("%20"," ")
        d = parse_qs(s1)
        predictorAllocator.deallocate(int(d["n"][0]))
        s+=" Predictor removed "+ d["n"][0]+"\n"
        ctype = 'text/html'
        s = s.replace("\n"," <br> ")
        s = s.replace("\r"," <br> ")
        response_body = '<html><body>' + s + '</body></html>'
    if environ['PATH_INFO'] == '/health':
        response_body = "1"
    if environ['PATH_INFO'] == '/test_get':
        s1 = environ['QUERY_STRING']
        s1 = s1.replace("%20"," ")
        d = parse_qs(s1)
        if "predict_list" in d:
            response_body = handle_predict_list(environ, predictorAllocator)
        else:
            response_body = '<html><body>' + s + '</body></html>'
    elif environ['PATH_INFO'] == '/env':
        response_body = ['%s: %s' % (key, value)
                    for key, value in sorted(environ.items())]
        response_body = '\n'.join(response_body)
    else:
        ctype = 'text/html'
#        response_body = '<html><body>' + s + '</body></html>'
        response_body = '''<form action="test_get" method="get" />
                            <input type="submit" value="predict_study" name="predict_study" />
                            <input type="submit" value="predict_create" name="predict_create" />
                            <input type="submit" value="predict" name="predict" />
                            <input type="submit" value="predict_remove" name="predict_remove" />
                            <input type="submit" value="predict_list" name="predict_list" />
                            </form>'''
    status = '200 OK'
    response_headers = [('Content-Type', ctype), ('Content-Length', str(len(response_body)))]
    #
    start_response(status, response_headers)
    predictorAllocator.save_to_file()
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
