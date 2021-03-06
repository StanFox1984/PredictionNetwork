#!/usr/bin/python
import os
import predict
from random import randint
from predict import run_all_tests
from predict import Predictor
from cgi import parse_qs
import sys
import difflib

from multiprocessing.managers import BaseManager
from multiprocessing import Process, Queue
import json
import pickle
import urllib
import urllib.request
import http
import http.cookies
#
# IMPORTANT: Put any additional includes below this line.  If placed above this
# line, it's possible required libraries won't be in your searchable path
#

#datadir         = os.environ['OPENSHIFT_DATA_DIR']
datadir = "./"
datafile_path   = datadir + "myfile"

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
      if os.path.isfile(datafile_path):
        f = open(datafile_path,'rb')
        self.predictor_array = pickle.load(f)
        f.close()
    def save_to_file(self):
      f = open(datafile_path,'wb')
      pickle.dump(self.predictor_array, f)
      f.close() # you can omit in most cases as the de
    def getArray(self):
      return self.predictor_array
    def getPredictor(self, n):
      try:
        return self.predictor_array[n]
      except KeyError:
        print ("No such predictor with id {0}", n)
        return None
    def deallocate(self, n):
      del self.predictor_array[n]
    def deallocate_all(self):
      self.predictor_array = {}

s = ""

class PredictorManager(BaseManager):
    pass


pmanager = None
predictorAllocator = PredictorAllocator(0,100)

class ClientState:
    def __init__(self):
        self.state = "DEFAULT"
        self.current_predictor_id = 0

client_id_to_state_dict = { }
last_client_id = 0

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
    script = '''
                //alert("predict_list!!!");
                function moveRight(){
                  document.body.style.left = parseInt(document.body.style.left) + 10 + 'px';
                }
                document.body.style.position= 'relative';
                document.body.style.left = '0px';
                //setInterval(moveRight, 1000);
             '''
    response_body = '<html><body style="background-color:powderblue;">' + s + '<script>\n'+script+'\n</script></body></html>'
    return response_body

def handle_predict_create(environ, predictorAllocator):
  s = ""
  if predictorAllocator != None:
#       s += str(os.getpid())
    s1 = environ['QUERY_STRING']
    s1 = s1.replace("%20"," ")
    d = parse_qs(s1)
#       s += str(d)
#       s += d["W"][0]
#    print d
    Wout = eval(d["W"][0])
    step = eval(d["step"][0])
    n = predictorAllocator.allocate(int(d["points_per_network"][0]), Wout, int(d["num_layers"][0]), step, int(d["max_iterations"][0]))

    if "weather_sample_alias" in d:
      p = predictorAllocator.getPredictor(n)
      p.set_alias("SUN_SHINE", 0)
      p.set_alias("RAIN", 1)
      p.set_alias("WARM", 2)
      p.set_alias("COLD", 3)
      p.set_alias("WINTER", 40)
      p.set_alias("SUMMER", 50)
      p.set_alias("GOOD_WEATHER", 6)
      p.set_alias("BAD_WEATHER", 7)

    if "stock_sample_alias" in d:
      p = predictorAllocator.getPredictor(n)
      p.set_alias("NASDAQ_DOWN", 0)
      p.set_alias("NASDAQ_UP", 1)
      p.set_alias("DOW_DOWN", 2)
      p.set_alias("DOW_UP", 3)
      p.set_alias("S&P_DOWN", 4)
      p.set_alias("S&P_UP", 5)
      p.set_alias("NYSE_DOWN", 6)
      p.set_alias("NYSE_UP", 7)

  return n

def handle_predict_study(environ, predictorAllocator):
  s = ""
  if predictorAllocator != None:
#   s += str(os.getpid())
    s1 = environ['QUERY_STRING']
    s1 = s1.replace("%20"," ")
    d = parse_qs(s1)
#   s += str(d)
#   s += d["W"][0]
    X = [ ]
    Y = [ ]
    n = eval(d["n"][0])
    _X = (d["X"][0])
    _Y = (d["Y"][0])
    if (_X.count('[') == 0):
      _X = eval('[ ' + _X + ' ]')
      X.append(_X)
    elif (_X.count('[') == 1):
      _X = eval(_X)
      X.append(_X)
    elif (_X.count('[') == 2):
      _X = eval(_X)
      X.extend(_X)
    if (_Y.count('[') == 0):
      _Y = eval('[ ' + _Y + ' ]')
      Y.append(_Y)
    elif (_Y.count('[') == 1):
      _Y = eval(_Y)
      Y.append(_Y)
    elif (_Y.count('[') == 2):
      _Y = eval(_Y)
      Y.extend(_Y)
    p = predictorAllocator.getPredictor(n)
    if p != None:
      p.study(X,Y)
      s+=" Predictor study "+ str(n)+"\n"
    else:
      s+=" Not found" + str(n)+"\n"
    ctype = 'text/html'
    s = s.replace("\n"," <br> ")
    s = s.replace("\r"," <br> ")
    response_body = '<html><body style="background-color:powderblue;">' + s + '</body></html>'
    return response_body

def handle_predict_study_from_link(environ, predictorAllocator):
  s = ""
  if predictorAllocator != None:
#   s += str(os.getpid())
    s1 = environ['QUERY_STRING']
    s1 = s1.replace("%20"," ")
    d = parse_qs(s1)
#   s += str(d)
#   s += d["W"][0]
    X = [ ]
    Y = [ ]
    if "predict_study_link_x" in d and "predict_study_link_y" in d:
      request = urllib.request.Request(d["predict_study_link_x"][0])
      response = urllib.request.urlopen(request)
      page = response.read()
#      print "X from page:", page
      X = eval(page)
      request = urllib.request.Request(d["predict_study_link_y"][0])
      response = urllib.request.urlopen(request)
      page = response.read()
      Y = eval(page)
#      print "Y from page:", page
    n = eval(d["n"][0])
    p = predictorAllocator.getPredictor(n)
    if p != None:
      p.study(X,Y)
      s+=" Predictor study "+ str(n)+"\n"
    else:
      s+=" Not found" + str(n)+"\n"
    ctype = 'text/html'
    s = s.replace("\n"," <br> ")
    s = s.replace("\r"," <br> ")
    response_body = '<html><body style="background-color:powderblue;">' + s + '</body></html>'
    return response_body

def handle_predict_set_alias(environ, predictorAllocator):
  s = ""
  if predictorAllocator != None:
#   s += str(os.getpid())
    s1 = environ['QUERY_STRING']
    s1 = s1.replace("%20"," ")
    d = parse_qs(s1)
#   s += str(d)
#   s += d["W"][0]
    n = eval(d["n"][0])
#    print d
    alias_key = (d["alias_key"][0])
    alias_value = (d["alias_value"][0])
#    print alias_key, alias_value
    key = alias_key
    value = alias_value
    p = predictorAllocator.getPredictor(n)
    if p != None:
      p.set_alias(key, value)
      s+=" Predictor set alias "+ str(n)+"\n"
    else:
      s+=" Not found" + str(n)+"\n"
    ctype = 'text/html'
    s = s.replace("\n"," <br> ")
    s = s.replace("\r"," <br> ")
    response_body = '<html><body style="background-color:powderblue;">' + s + '</body></html>'
    return response_body

def handle_predict(environ, predictorAllocator):
  s= ""
  if predictorAllocator != None:
    #s += str(os.getpid())
    s1 = environ['QUERY_STRING']
    s1 = s1.replace("%20"," ")
    d = parse_qs(s1)
#    s += str(d)
#    s += d["W"][0]
    X = [ ]
    n = eval(d["n"][0])
    depth = eval(d["depth"][0])
    _X = (d["X"][0])
    if (_X.count('[') == 0):
      _X = eval('[ ' + _X + ' ]')
      X.append(_X)
    elif (_X.count('[') == 1):
      _X = eval(_X)
      if len(_X) > 0:
        X.append(_X)
    elif (_X.count('[') == 2):
      _X = eval(_X)
      X.extend(_X)
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
    response_body = '<html><body style="background-color:powderblue;">' + s + '</body></html>'
    return response_body

def handle_predict_remove(environ, predictorAllocator):
  s = ""
  s1 = environ['QUERY_STRING']
  s1 = s1.replace("%20"," ")
  d = parse_qs(s1)
  predictorAllocator.deallocate(int(d["n"][0]))
  s+=" Predictor removed "+ d["n"][0]+"\n"
  ctype = 'text/html'
  s = s.replace("\n"," <br> ")
  s = s.replace("\r"," <br> ")
  response_body = '<html><body style="background-color:powderblue;">' + s + '</body></html>'
  return response_body

def handle_predict_remove_all(environ, predictorAllocator):
  s = ""
  s1 = environ['QUERY_STRING']
  s1 = s1.replace("%20"," ")
  d = parse_qs(s1)
  predictorAllocator.deallocate_all()
  s+="All predictors removed\n"
  ctype = 'text/html'
  s = s.replace("\n"," <br> ")
  s = s.replace("\r"," <br> ")
  response_body = '<html><body style="background-color:powderblue;">' + s + '</body></html>'
  return response_body

def handle_run_tests(environ):
  s = ""
  s += predict.run_all_tests()
  s = s.replace("\n"," <br> ")
  s = s.replace("\r"," <br> ")
  ctype = 'text/html'
  s = s.replace("\n"," <br> ")
  s = s.replace("\r"," <br> ")
  response_body = '<html><body style="background-color:powderblue;">' + s + '</body></html>'
  return response_body

def application(environ, start_response):
    global s
    global last_client_id
    global client_id_to_state_dict
    cur_client_id = 0


    client_state = None
    if 'HTTP_COOKIE' in environ:
        s1 = environ['HTTP_COOKIE']
        print ("Got cookie:", s1)
        s1 = s1.replace("%20"," ")
#        d = parse_qs(s1)
        d = { }
        l = difflib.get_close_matches("last_client_id", s1.replace(" ", "").split(";"))[0].split("=")
        d[l[0]] = l[1]
        print (d)
        if not "last_client_id" in d:
            cur_client_id = last_client_id
            last_client_id += 1
            print ("New client!", cur_client_id)
            client_state = ClientState()
            client_id_to_state_dict[str(cur_client_id)] = client_state
        else:
            cur_client_id = int(d["last_client_id"])
            print ("Known client! ", cur_client_id)
            print (client_id_to_state_dict)
            if not str(cur_client_id) in client_id_to_state_dict:
                print ("Not found(bug?)")
                client_state = ClientState()
                client_id_to_state_dict[str(cur_client_id)] = client_state
            client_state = client_id_to_state_dict[str(cur_client_id)]
            print ("Client state: %s Predictor id %d" % (client_state.state, client_state.current_predictor_id))
    else:
        cur_client_id = last_client_id
        last_client_id += 1
        print ("New client!", cur_client_id)
        client_state = ClientState()
        client_id_to_state_dict[str(cur_client_id)] = client_state
    ctype = 'text/html'
    s = ""
    s1 = ""
    response_body = ""
    predictorAllocator.load_from_file()
    aliases = None
    n = None
    if environ['PATH_INFO'] == '/tests':
        response_body = handle_run_tests(environ)
    if environ['PATH_INFO'] == '/predict_list':
        response_body = handle_predict_list(environ, predictorAllocator)
    if environ['PATH_INFO'] == '/predict_create':
        response_body = handle_predict_create(environ, predictorAllocator)
    if environ['PATH_INFO'] == '/predict_study':
        response_body = handle_predict_study(environ, predictorAllocator)
    if environ['PATH_INFO'] == '/predict_study_from_link':
        response_body = handle_predict_study_from_link(environ, predictorAllocator)
    if environ['PATH_INFO'] == '/predict':
        response_body = handle_predict(environ, predictorAllocator)
    if environ['PATH_INFO'] == '/predict_remove':
        response_body = handle_predict_remove(environ, predictorAllocator)
    if environ['PATH_INFO'] == '/health':
        response_body = "1"
    if environ['PATH_INFO'] == '/test_get':
        s1 = environ['QUERY_STRING']
        s1 = s1.replace("%20"," ")
        d = parse_qs(s1)
#        print d
        c = 0
        if "predict_list" in d:
            response_body = handle_predict_list(environ, predictorAllocator)
            c = 1
        if "predict_create" in d:
            n = handle_predict_create(environ, predictorAllocator)
            client_state.state = "SETALIAS"
            _s = ""
            _s+=" Predictor created "+ str(n)+"\n"
            ctype = 'text/html'
            _s = _s.replace("\n"," <br> ")
            _s = _s.replace("\r"," <br> ")
            response_body = '<html><body style="background-color:powderblue;">' + _s + '</body></html>'
            c = 1
        if "predict_study" in d:
            response_body = handle_predict_study(environ, predictorAllocator)
            client_state.state = "PREDICT"
            c = 1
        if "predict_study_from_link" in d:
            response_body = handle_predict_study_from_link(environ, predictorAllocator)
            client_state.state = "PREDICT"
            c = 1
        if "predict" in d:
            response_body = handle_predict(environ, predictorAllocator)
            c = 1
        if "home" in d:
            response_body = ""
            client_state.state = "DEFAULT"
        if "predict_remove" in d:
            response_body = handle_predict_remove(environ, predictorAllocator)
            c = 1
        if "predict_remove_all" in d:
            response_body = handle_predict_remove_all(environ, predictorAllocator)
            c = 1
        if "predict_set_alias" in d:
            response_body = handle_predict_set_alias(environ, predictorAllocator)
            client_state.state = "STUDY"
            c = 1
        if "predict_run_tests" in d:
            response_body = handle_run_tests(environ)
            c = 1
        if c == 0:
            response_body = '<html><body style="background-color:powderblue;">' + s + '</body></html>'
    elif environ['PATH_INFO'] == '/env':
        response_body = ['%s: %s' % (key, value)
                    for key, value in sorted(environ.items())]
        response_body = '\n'.join(response_body)
    elif environ['PATH_INFO'] == '/X_Input':
        f = open(datadir+"X_Input", "rb")
        fl = f.read()
        response_body = fl
        status = '200 OK'
        response_headers = [('Content-Type', ctype), ('Content-Length', str(len(response_body)))]
        start_response(status, response_headers)
        return [ response_body ]
    elif environ['PATH_INFO'] == '/Y_Input':
        f = open(datadir+"Y_Input", "rb")
        fl = f.read()
        response_body = fl
        status = '200 OK'
        response_headers = [('Content-Type', ctype), ('Content-Length', str(len(response_body)))]
        start_response(status, response_headers)
        return [ response_body ]
    query_dict = parse_qs(environ['QUERY_STRING'])
    if "n" in query_dict:
        if query_dict["n"][0] != "predictor_id":
          p = predictorAllocator.getPredictor(int(query_dict["n"][0]))
          if p != None:
            client_state.current_predictor_id = int(query_dict["n"][0])
            aliases = p.get_aliases()

    if n != None:
      client_state.current_predictor_id = n
      p = predictorAllocator.getPredictor(n)
      if p != None:
        aliases = p.get_aliases()

    ctype = 'text/html'

    if len(response_body) == 0:
      response_body = '<html><body style="background-color:powderblue;">' + '</body></html>'
    select_s = '''                  select = document.getElementById("aliases");
               '''
    if aliases != None:
        for key in aliases:
            select_s += "     var el = document.createElement(\"option\"); " +\
                                "el.textContent = \'"+key+"\';"+\
                                "el.value = \'"+key+"\';"+\
                                "select.appendChild(el);"

    if client_state.state == "DEFAULT":
        with open('MainView.htm') as f:
            read_data = f.read()
    if client_state.state == "STUDY":
        with open('MainViewStudy.htm') as f:
            read_data = f.read()
    if client_state.state == "PREDICT":
        with open('MainViewPredict.htm') as f:
            read_data = f.read()
    if client_state.state == "CREATE":
        with open('MainViewCreate.htm') as f:
            read_data = f.read()
    if client_state.state == "SETALIAS":
        with open('MainViewSetAlias.htm') as f:
            read_data = f.read()

    response_body += read_data + select_s + "</script>"
    status = '200 OK'
    ctype += ";charset=utf-8"
    response_headers = [('Content-Type', ctype), ('Content-Length', str(len(response_body))), ('Set-Cookie', "last_client_id=" + str(cur_client_id))]
    #
    start_response(status, response_headers)
    predictorAllocator.save_to_file()
    if type(response_body) == str:
        return [ response_body.encode("utf-8") ]
    return response_body

if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    httpd = make_server('localhost', int(sys.argv[1]), application)
    httpd.serve_forever()
