#!/usr/bin/env python

import time
import sys
import random
import copy
import math
import os
import commands
import difflib
from threading import Thread
from multiprocessing.managers import BaseManager
from multiprocessing import Pool
from tinythreadpool import TinyThreadPool
from StringIO import StringIO
import sys

class ProbTree:
    probnodes = {}
    def __init__(self):
        self.outcomes = {}
        self.pass_by_hits = { }
        self.endpoint_hits = { }
        self.data = None
        self.me = None
        self.probs = { }
        self.r1 = { }
        self.r2 = { }
        self.stamp = time.time()
        self.total_hits = 0
    def _addOutcome_Internal(self, key, data, last = False):
        p = None
        if key not in self.outcomes:
#            print key," not in ", self.outcomes
            if key not in ProbTree.probnodes:
                p = ProbTree()
                p.data = data
                p.me = ( key, data)
                ProbTree.probnodes[key] = p
            else:
                p = ProbTree.probnodes[key]
            self.outcomes[key] = p
            self.pass_by_hits[key] = 0
            self.endpoint_hits[key] = 0
#            print "Adding ", key, " to ", self.me
        else:
#            print key," found in ", self.outcomes
            p = self.outcomes[key]
        if last == False:
            self.pass_by_hits[key] += 1
#            print key," not last incremented pass_by:",self.pass_by_hits[key]
        else:
            self.endpoint_hits[key] += 1
#            print key," is last incremented endpoint:",self.endpoint_hits[key]
        self.total_hits += 1
#        print self.pass_by_hits
        #p.probability = float(p.pass_by_hits + p.endpoint_hits)/float(self.pass_by_hits )

        return p
    def _recalcProbs(self):
        last_prob = 0.0
#        print "recalc ", self.me, self.outcomes
        for p in self.outcomes:
          self.probs[p] = float(self.pass_by_hits[p] + self.endpoint_hits[p])/float(self.total_hits )
          self.r1[p] = last_prob
          self.r2[p] = last_prob + self.probs[p]
          last_prob = self.r2[p]
    def recalcProbs(self, req_stamp = None):
        if req_stamp != None:
            if self.stamp == req_stamp:
                return
            self.stamp = req_stamp
        else:
            self.stamp = time.time()
        self._recalcProbs()
        for p in self.outcomes:
          self.outcomes[p].recalcProbs(self.stamp)
    def addOutcome(self, in_arr):
        tmp = self
        i = 0
        for el in in_arr:
          if i == len(in_arr)-1:
            last = True
          else:
            last = False
          tmp = tmp._addOutcome_Internal(el[0], el[1], last)
          i += 1
        self.recalcProbs()
    def printOutcomes(self, key = None, req_stamp = None):
        if req_stamp != None:
            if self.stamp == req_stamp:
                return
            self.stamp = req_stamp
        else:
            self.stamp = time.time()
#        print self.me, "Pass by hits: ", self.pass_by_hits," Endpoint hits: ",self.endpoint_hits," Key:", key, " Data:", self.data
#        print "Probs:"
    #    for el in self.outcomes:
#          print self.probs[el], self.outcomes[el].me
        for el in self.outcomes:
          self.outcomes[el].printOutcomes(el, self.stamp)
        print "*************************"

    def generateWithProb(self, in_prefix, max_len):
        ret = [ ]
        tmp = self
        i = 0
        while True:
          if i >= len(in_prefix):
            break
          if in_prefix[i] not in tmp.outcomes:
            break
          tmp = tmp.outcomes[in_prefix[i]]
#          print tmp.me
          i += 1
        if len(in_prefix) != i:
#            print in_prefix, i, tmp.outcomes
            return None
        while True:
          r = random.random()
          if len(tmp.outcomes) == 0:
              break
          for el in tmp.outcomes:
            if r < tmp.r2[el] and r >= tmp.r1[el]:
                ret.append((el, tmp.outcomes[el].data))
                if len(ret) > max_len:
                    return ret
                if len(tmp.outcomes[el].outcomes) > 0:
                    tmp = tmp.outcomes[el]
#                    print tmp.me
                    break
                else:
                    return ret
        return ret

class ProbNetwork:
    def __init__(self):
        pass
    def addAssociativeChain(self, chain):
        vec = [ ( el.key, el ) for el in chain ]
        self.addOutcome(vec)
    def addOutcome(self, in_arr):
        p = None
#        print in_arr
        if in_arr[0][0] in ProbTree.probnodes:
            p = ProbTree.probnodes[in_arr[0][0]]
        else:
            p = ProbTree()
            p.me = in_arr[0]
            p.data = in_arr[0][1]
            ProbTree.probnodes[in_arr[0][0]] = p
        p.addOutcome(in_arr[1:])
    def printOutcomes(self, key = None):
        if key == None:
            for el in ProbTree.probnodes:
                ProbTree.probnodes[el].printOutcomes(el)
        else:
            if key in ProbTree.probnodes:
                ProbTree.probnodes[key].printOutcomes()
    def generateWithProb(self, in_prefix, max_len):
        vec = [ el.key for el in in_prefix ]
#        print vec
        res = self._generateWithProb(vec, max_len)
#        print res
        out = [ ProbTree.probnodes[el[0]].data for el in res ]
        return out
    def _generateWithProb(self, in_prefix, max_len):
        if in_prefix[0] in ProbTree.probnodes:
#            print ProbTree.probnodes[in_prefix[0]].me
            return ProbTree.probnodes[in_prefix[0]].generateWithProb(in_prefix[1:], max_len)

class Entity:
    def __init__(self, key, data):
        self.key = key
        self.data = data
    def __str__(self):
        return str(self.key) + "" +  str(self.data)
    @staticmethod
    def generateEntity(key):
        return Entity(key, "Entity" + str(key))

def getVecDiffAbs(str1, str2):
    n = min([len(str1), len(str2)])
    res = 0
    for i in xrange(0, n):
      ch1 = str1[i]
      ch2 = str2[i]
      res += abs(int(ch1)-int(ch2))
    return res

def getVecDiff(str1, str2):
    n = min([len(str1), len(str2)])
    res = 0
    for i in xrange(0, n):
      ch1 = str1[i]
      ch2 = str2[i]
      res += int(ch1)-int(ch2)
    return res

def getVecAbs(vec):
    n = 0
    for v in vec:
      n += abs(v)
    return n

def getArrCorrelations( X, Y):
    prev_X = None
    prev_Y = None
    Corr = [ ]
    for j in xrange(0, len(X[0])):
      Corr.append(0)
    for i in xrange(0, len(X)):
      if prev_X != None:
        for j in xrange(0, len(X[i])):
          if prev_X[j] != X[i][j]:
            Corr[j] += getVecDiffAbs(prev_Y, Y[i])
      prev_X = X[i]
      prev_Y = Y[i]
    return Corr

def getVecDiffAbsWithCorr(str1, str2, corr):
    norm = max(corr)
    _corr = [ float(c)/float(norm) for c in corr ]
    n = min([len(str1), len(str2)])
    res = 0
    for i in xrange(0, n):
      ch1 = str1[i]
      ch2 = str2[i]
      res += _corr[i]*(abs(int(ch1)-int(ch2)))
    return res

class BitArray:
    def __init__(self, initial_bits = 32, int_size = 32):
        self.array = [ 0 for i in xrange(0, (initial_bits / int_size) ) ]
        self.int_size = int_size
    def getBit(self, bit):
        octet = bit / self.int_size
        offset = bit % self.int_size
        if octet > len(self.array):
            self.array.extend( [ 0 for i in xrange(octet  - len(self.array)) ] )
        n = 1 << offset
        if self.array[octet] & n:
            return 1
        return 0
    def getNumber(self, _offset, num_bits = None):
        if num_bits == None:
            num_bits = self.int_size
        octet = _offset / self.int_size
        offset = _offset % self.int_size
        if octet > len(self.array):
            self.array.extend( [ 0 for i in xrange(octet  - len(self.array)) ] )
        shift = 0
        number = 0
        n = 1 << offset
        while shift < num_bits:
            if n & self.array[octet]:
                number |= n
            n = n << 1
            shift += 1
        number = number >> offset
        return number
    def setBit(self, bit, value):
        octet = bit / self.int_size
        offset = bit % self.int_size
        if octet > len(self.array):
            self.array.extend( [ 0 for i in xrange(octet  - len(self.array)) ] )
        shift = 0
        n = value
        while shift != offset:
            n = n << 1
            shift += 1
        if n == 1<<shift:
          self.array[octet] |= n
        if n == 0:
          self.array[octet] &= ~n
    def setArray(self, arr):
        self.array = [ ]
        for a in arr:
          b = BitArray()
          b.setNumber(0, int(a))
          self.extendWith(b)
    def setNumber(self, _offset, value, bits_in_value = None):
        if bits_in_value == None:
            bits_in_value = self.int_size
        octet = _offset / self.int_size
        offset = _offset % self.int_size
        if octet > len(self.array):
            self.array.extend( [ 0 for i in xrange(octet  - len(self.array)) ] )
        shift = 0
        n = value
        while shift != offset:
            n = n << 1
            shift += 1
        i = 1<<shift
        bits = 0
        while bits < bits_in_value:
            if i & n:
                self.array[octet] |= i
            else:
                self.array[octet] &= ~i
            i = i<<1
            bits += 1
    def extendWith(self, bit_array):
        self.array.extend(bit_array.array)
    def copyFrom(self, bit_array):
        self.array = [ ]
        self.array.extend(bit_array.array)
        self.int_size = bit_array.int_size
    def printBits(self, normal = False):
        for i in reversed(self.array):
          if normal == False:
            shift = pow(2, self.int_size)
            while shift != 0:
                if shift & i:
                    sys.stdout.write("1")
                else:
                    sys.stdout.write("0")
                sys.stdout.flush()
                shift = shift>>1
          if normal == True:
            sys.stdout.write(str(i))
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    def getAsOne(self):
        s = ""
        for i in reversed(self.array):
            s += str(i)
        return int(s)

class Factor:
    def __init__(self, offset, width, input_function):
        self.offset = offset
        self.width = width
        self.input_function = input_function
    def invokeInput(self):
        res = self.input_function()
        return res

class FactorManager:
    def __init__(self):
        self.factors = [ ]
        self.bit_array = BitArray()
        self.last_offset = 0
    def addFactor(self, width, input_function):
        self.factors.append(Factor(self.last_offset, width, input_function))
        self.last_offset += width
    def refillState(self):
        for factor in self.factors:
            self.bit_array.setNumber(factor.offset, factor.invokeInput(), factor.width)
        return self.bit_array.getAsOne()





SUN=1
RAIN=2
SNOW=4
HOT=8
COLD=16
wi = 0
total = 0
probs = 0

def _distance_func(factor1, factor2):
    seq=difflib.SequenceMatcher(a=str(factor1).lower(), b=str(factor2).lower())
    r = int(seq.ratio()*100.0)
    return r


def generate_entity_weather(factor):
    global wi
    global total
    global probs
    if wi == 0:
      wi = [ 1 for i in xrange(0,32) ]
    if probs == 0:
      probs = [ 0 for i in xrange(0,32) ]
    total += 1
    s = ""
    if factor & SUN:
        s += "SUN "
    if factor & RAIN:
        s += "RAIN "
    if factor & SNOW:
        s += "SNOW "
    if factor & HOT:
        s += "HOT "
    if factor & COLD:
        s += "COLD "
    return Entity(factor, s)

def generate_entity_x(factor):
    return Entity(factor, factor)

class FactorAnalyzer:
    def __init__(self):
        self.factor_map = { }
        self.prob_network = ProbNetwork()
        self.last_state = None
        self.total_factor_hits = 0
        self.corr = [ ]
        self.factor_outcome_list = [ ]
    def analyze_factor(self, factor, generate_func = None, outcome = None):
        state = None
        if factor not in self.factor_map:
            if generate_func == None:
              state = Entity.generateEntity(factor)
            else:
              state = generate_func(factor)
            self.factor_map[factor] = state
#            print "Added new factor ", factor, state
        else:
            state = self.factor_map[factor]
        if self.last_state == None:
            self.last_state = state
        else:
            self.prob_network.addAssociativeChain([ self.last_state, state])
            self.last_state = state
        if outcome != None:
          self.factor_outcome_list.append((factor, outcome))
          self.corr = getArrCorrelations([ eval(r[0]) for r in self.factor_outcome_list ], [ eval(r[1]) for r in self.factor_outcome_list ])
    def deduct(self, prefix_factors, depth, generate_func = None, distance_func = _distance_func):
        distance_metric = 0
        closest_factor  = None
        prefix_states = [ ]
        i = 0
        for factor in prefix_factors:
            if factor not in self.factor_map:
#                print factor, "not in ", self.factor_map
                for el in self.factor_map:
#                    print el
                    if len(self.factor_outcome_list) > 0:
                      d = getVecDiffAbsWithCorr(eval(el), eval(factor), self.corr)
                    else:
                      if distance_func != None:
                        d = distance_func(el, factor)
                      else:
                        return None
                    if closest_factor == None or d < distance_metric:
                        closest_factor = el
                        distance_metric = d
            else:
                closest_factor = factor
            prefix_factors[i] = closest_factor
            i += 1
            if generate_func == None:
              prefix_states.append(Entity.generateEntity(closest_factor))
            else:
              prefix_states.append(generate_func(closest_factor))
        return self.prob_network.generateWithProb(prefix_states, depth)

def sunny():
    return 1

def rain_or_not():
    return 1

#Yj = sum (Xij Wj)
#Err func(Wj) = sum_m( Ymj - sum_i(XmiWj) )^2    j = 0..len(X), 0..len(Y), i = range(j), m= 0..len(observations)
#gradient(Err_func(Wj), j) = sum_m ( -2 * Ymj * sum_i(Xmi) + sum_i(Xmi*2*Wj))

#Err func(Wj) = ( Yj - sum_i(XiWj) )^2    j = 0..len(X), 0..len(Y), i = range(j), m= 0..len(observations)

#Err func2(Wj) = sum_m( Ymj - sum_i(XmiWj) )^2    j = 0..len(X), 0..len(Y), i = range(j), m= 0..len(observations)
#gradient(Err_func2(Wj), j) =  -1 * sum_m(-2 * Ymj * sum_i(Xi) + sum_i(Xmi*2*Wj))

#d(Yj^2 - 2 Yj sum_i(XiWj) + (sum_i(XiWj)^2)/dWj = -2 Yj sum_i(Xi) + sum_i(Xi^2*2*Wj)



class NeuralLinearLayer:
    def __init__(self, W0, step = None, max_iterations = 10):
      self.W = list(W0)
      self.last_err_func = None
      if step:
        self.step = list(step)
      else:
        self.step = [ 0.001 for n in self.W ]
      self.max_iterations = max_iterations
    def study(self, X, Y):
      grad = [ 0 for n in self.W ]
      Wout = list(self.W)
      for i in xrange(0, len(Y)):
        iterations = 0
        self.last_err_func = None
        while True and iterations < self.max_iterations:
#          print "Starting weights: " , self.W
          self.calc_gradient(X[i], Y[i], grad, Wout)
          n = self.calc_err_func(X[i], Y[i])
          if self.last_err_func == None or n < self.last_err_func:
            self.last_err_func = n
          else:
#            print "Last error ", n," is more than prev ", self.last_err_func
            break
          self.W = list(Wout)
          iterations += 1
#          print "W:", Wout
#          print "Gradient vector:", grad
#          print "Err:", n
#          print "Step:", self.step

    def study2(self, X, Y):
      grad = [ 0 for n in self.W ]
      Wout = list(self.W)
      iterations = 0
      overall_iterations = 0
      self.last_err_func = None
      self.n = None
      grad_zero = True
#      print "Initial err: ", self.calc_err_func2(X, Y), "W: ", self.W, " max_iterations: ", self.max_iterations, " step: ", self.step
      iterations_left = self.max_iterations
      while True and iterations < iterations_left:
#         print "Starting weights: " , self.W
#        print "Calculating gradient and weights for ", X, Y
        if iterations >= (iterations_left/4):
          for l in xrange(0, len(self.step)):
            self.step[l] *= 2
          overall_iterations += iterations
          iterations_left -= iterations
#          print "Increased step to ", self.step, "iterations_left: ", iterations_left, "W: ", self.W, "last_grad: ", grad
          for h in xrange(0, len(X)):
            _Y = copy.deepcopy(Y)
            t = self.calc_y(X[h], _Y[h])
            _Y[h] = copy.deepcopy(t)
#            print "X: ", X[h], "Y: ", _Y[h]
          iterations = 0
        g = self.calc_gradient2(X, Y, grad)
        grad = copy.deepcopy(g)
        for j in xrange(0, len(self.W)):
          Wout[j] = self.W[j]
          self.W[j] += self.step[j]*grad[j]
        n = self.calc_err_func2(X, Y)
        for h in grad:
          if h != 0:
            grad_zero = False
            break
        if grad_zero == True:
#          print "Grad is zero: ", grad
          if n == 0:
            break
          else:
            if len(X) > 2:
              break
            else:
              for l in xrange(0, len(self.step)):
                self.step[l] *= 8
#              print "Probable local maximum detected, drastically increasing step to ", self.step
              for j in xrange(0, len(self.W)):
                Wout[j] = self.W[j]
                self.W[j] += self.step[j]*grad[j]
                n = self.calc_err_func2(X, Y)
        if self.last_err_func == None or n <= self.last_err_func:
          self.last_err_func = n
        else:
#          print "Last error ", n," is more than prev ", self.last_err_func
          for j in xrange(0,len(self.W)):
            self.W[j] = Wout[j]
          break
        iterations += 1
      overall_iterations += iterations
#      print "Final err: ", self.calc_err_func2(X, Y), "W: ", self.W, " after ", overall_iterations, " iterations", X, Y, grad
#      print "Next error was: ", n
#          print "W:", Wout
#          print "Gradient vector:", grad
#        print "Err:", n
#          print "Step:", self.step

    def calc_y(self, X, Y):
        for j in xrange(0, len(Y)):
          Y[j] = 0.0
          for i in xrange(0, len(X)):
            Y[j]+=(self.W[j])*X[i]
        return Y

    def calc_gradient2(self, X, Y, grad):
      for j in xrange(0, len(self.W)):
        grad[j] = 0
        sum_i = 0
        sum_i_2 = 0
        for m in xrange(0, len(Y)):
          for i in xrange(0, len(X[m])):
#            sum_i += X[m][i]
#            sum_i_2 += X[m][i]*X[m][i]
            if ((2.0 * Y[m][j] * X[m][i] - X[m][i]*X[m][i]*2.0*self.W[j])) != 0:
              grad_before = grad[j]
              grad[j] += ((2.0 * Y[m][j] * X[m][i] - X[m][i]*X[m][i]*2.0*self.W[j]))#/abs((2.0 * Y[m][j] * X[m][i] - X[m][i]*X[m][i]*2.0*self.W[j]))
#              if grad[j] == 0:
#                print "Grad is zero: 2.0 * Y[m][j] * X[m][i] = ", 2.0 * Y[m][j] * X[m][i], " X[m][i]*X[m][i]*2.0*self.W[j] = ", X[m][i]*X[m][i]*2.0*self.W[j], grad_before
#            else:
#              print "Grad is zero: 2.0 * Y[m][j] * X[m][i] = ", 2.0 * Y[m][j] * X[m][i], " X[m][i]*X[m][i]*2.0*self.W[j] = ", X[m][i]*X[m][i]*2.0*self.W[j]
      for j in xrange(0, len(self.W)):
        if grad[j] != 0:
          grad[j] = grad[j]/abs(grad[j])
      return grad
#        print "W[j](", Wout[j],")+=", self.step[j]*grad[j]

    def calc_err_func(self, X, Y):
      Err = 0
      for j in xrange(0, len(Y)):
        s = 0.0
        for i in xrange(0, len(X)):
            s += X[i]*self.W[j]
        Err += pow(( Y[j] - s ), 2)
      return Err

    def calc_err_func2(self, X, Y):
      Err = 0
      for m in xrange(0, len(Y)):
        for j in xrange(0, len(Y[m])):
          s = 0.0
          for i in xrange(0, len(X[m])):
            s += X[m][i]*self.W[j]
          Err += pow(( Y[m][j] - s ), 2)
      return Err




class NeuralLinearNetwork:
    def __init__(self, W0, num_layers, step = None, max_iterations = 10, sp = 1):
      self.layers = [ ]
      self.stop_layer = None
      W = W0
      for i in xrange(0, num_layers):
          self.layers.append(NeuralLinearLayer(W, step, max_iterations))
      self.sp = sp
      self.last_yout = None
    def study(self, X, Y):
      Y0 = copy.deepcopy(Y)
      X0 = copy.deepcopy(X)
      X_layers = [ ]
      Out = copy.deepcopy(Y)
      for m in xrange(0, len(Out)):
        for j in xrange(0, len(Out[m])):
          Out[m][j] = 0
      for i in xrange(0, len(self.layers)):
  #        for j in reversed(xrange(0, i)):
  #        for m in xrange(0, len(X0)):
  #          X0[m].extend(X_layers[j][m])
#        for m in xrange(0, len(Y0)):
#          print os.getpid(), ": Teaching layer ", i, " X0:",X0[m]," Y0:", Y0[m]
        self.layers[i].study2(X0, Y0)
        X_layers.append(X0)
#       print X0, Y0
        for m in xrange(0, len(Y0)):
          YY = self.layers[i].calc_y(X0[m], Y0[m])
          for j in xrange(0, len(Y0[m])):
            Out[m][j] += Y0[m][j]
#          print "Got approximation: X0:",X0[m]," Out:", Out[m], "Y0: ", Y0[m]
#       print X0, Y0
        X0 = copy.deepcopy(Out)
        for m in xrange(0, len(Y0)):
          for j in xrange(0, len(Y0[m])):
            Y0[m][j] = Y[m][j] - Out[m][j]

      self.stop_layer = None
      Err = [ ]
      Y0 = [ 0 for i in xrange(0, len(Y[0])) ]
      for i in xrange(0, len(self.layers)):
        s = 0
        for m in xrange(0, len(Y)):
          YY = self.calc_y2(X[m], Y0, i + 1)
          for y in xrange(0, len(Y0)):
            Y0[y] = copy.deepcopy(YY[y])
          for j in xrange(0, len(Y[m])):
            s += abs(Y0[j] - Y[m][j])
        Err.append(s)
#        print "Layer ", i," err: ", s

      self.stop_layer = Err.index(min(Err)) + 1
#      print os.getpid(),": Stop layer: ", self.stop_layer

    def getWeights(self, layer):
      return self.layers[layer].W

    def calc_y2(self, X, Y, up_to = None):
      X0 = list(X)
      Y0 = list(Y)
      Out = [ 0 for n in xrange(0, len(Y)) ]
#      print os.getpid(), ": Stop layer for calc: ", self.stop_layer
      if self.stop_layer != None:
        max_layer = self.stop_layer if self.stop_layer < len(self.layers) else len(self.layers)
      else:
        max_layer = len(self.layers)
        if up_to != None:
          max_layer = up_to if up_to < len(self.layers) else len(self.layers)
      for i in xrange(0, max_layer):
#        print "Layer ", i
        YY = self.layers[i].calc_y(X0, Y0)
        for y in xrange(0, len(Y0)):
          Y0[y] = copy.deepcopy(YY[y])
        for j in xrange(0, len(Out)):
#          print Out[j]," + ", Y0[j]
          Out[j] += Y0[j]
        X0 = list(Out)
  #      X0.extend(list(Y0))
      for i in xrange(0, len(Out)):
        Y[i] = Out[i]
#      print "Res:", Y, X
      return Y

class NeuralNetworkManager(BaseManager):
    pass

NeuralNetworkManager.register('Network', NeuralLinearNetwork)

def createNetworkForNode(node_addr, _authkey, W0, num_layers, step = None, max_iterations = 10):
    m = NeuralNetworkManager(address=node_addr, authkey=_authkey)
#    print m
    m.connect()
    network = m.Network(W0, num_layers, step, max_iterations)
    return network


class NeuralLinearComposedNetwork:
    def __init__(self, points_per_network, W0, num_layers, step = None, max_iterations = 10, parallelize = False, sp = 1):
      self.networks = [ ]
      self.W0 = W0
      self.num_layers = num_layers
      self.step = step
      self.parallelize = parallelize
      self.max_iterations = max_iterations
      self.sp = sp
      self.points_per_network = points_per_network
      self.cyclic  = [ False for j in self.W0 ]
      self.mn = [ [ None ] for j in xrange(0, len(W0)) ]
      self.mx = [ [ None ] for j in xrange(0, len(W0)) ]
      self.pool = None
      self.acc_x = [ ]
      self.acc_y = [ ]

    def autoCorrelation( self, Y, n ):
      Corr = 0
      for m in xrange(0, len(Y)):
        Corr+=abs((Y[m][0] - Y[(m+n) % len(Y)][0]))
      return Corr

    def detectPeriodic( self, Y ):
      prev_corr = None
      Corr = None
      min_corr = None 
      num_periods = 0
      df = 0
      prev_df = 0
      for i in xrange(0, len(Y)/2):
#        print i
        prev_corr = Corr
        Corr = self.autoCorrelation( Y, i )
#        print "Corr:",Corr
        if prev_corr != None:
          prev_df = df
          df = abs(prev_corr - Corr)
#          print df
          if prev_corr > Corr:
            num_periods += 1
            if min_corr == None or min_corr[0] > Corr:
              min_corr = ( Corr, i )
        else:
          prev_corr = Corr
#          print Corr
      if min_corr != None and num_periods >= 1:
#        print "Detect periodic for ", Y
#        print "Periodic detected maximum correlation at ", min_corr[1], Y[min_corr[1]]
        return True
#      print "Detect periodic for ", Y
      return False

    def _detectPeriodic( self, Y ):
      prev_corr = None
      Corr = None
      min_corr = None 
      num_periods = 0
      for i in xrange(0, len(Y)/2):
        prev_corr = Corr
        Corr = self.autoCorrelation( Y, i )
        if prev_corr != None:
          if prev_corr > Corr:
            num_periods += 1
            if min_corr == None or min_corr[0] > Corr:
              min_corr = ( Corr, i )
        else:
          prev_corr = Corr
#        print Corr
      if min_corr != None and num_periods >= 1:
#        print "Detect periodic for ", Y
#        print "Periodic detected maximum correlation at ", min_corr[1], Y[min_corr[1]]
        return True
#      print "Detect periodic for ", Y
      return False

    def nstudy_wrapper(self, network, X, Y):
      network.study(X, Y)

    def study(self, X, Y):
      self.acc_y.extend(Y)
      for j in xrange(0, len(self.W0)):
#        if not self.detectPeriodic( [ [ X[m][j] ] for m in xrange(0, len(X)) ] ):
         self.cyclic[j] = self.detectPeriodic( [ [ self.acc_y[m][j] ] for m in xrange(0, len(self.acc_y)) ] )
#      print self.cyclic
      n = 0
      added_X = [ ]
      added_Y = [ ]
      for m in xrange(0, len(X)):
        added_X.append( X[m] )
        added_Y.append( Y[m] )
        n+=1
        if n >= self.points_per_network or m == len(X)-1:
          n = 0
          _added_X = [ [ u for u in xrange(0, len(added_X))] for u1 in xrange(0, len(self.W0)) ]
          for j in xrange(0, len(self.W0)):
            for _m in xrange(0, len(added_X)):
              _added_X[j][_m]= added_X[_m][j]
#          print _added_X
          mn = [ min(x) for x in _added_X ]
          mx = [ max(x) for x in _added_X ]

          found = False
          ii = 0
          for network in self.networks:
            ii += 1
            if network[0] <= mn and network[1] >= mx:
              if self.parallelize == False:
                network[2].study(added_X, added_Y)
              else:
                if self.pool == None:
                  self.pool = TinyThreadPool(10)
                  self.pool.start()
#                print "enqueue"
                self.pool.enqueue_task_id(ii % 10, NeuralLinearComposedNetwork.nstudy_wrapper,self, network[2], added_X, added_Y)
#              print "Updated ", mn, mx, added_X, added_Y
              found = True
              break
          if found == False:
#             self.networks[j].append( ( mn, mx, NeuralLinearNetwork( [self.W0[j] for u in xrange(0, min([self.points_per_network, len(added_X)])) ], self.num_layers, [ self.step[0] for u in xrange(0, min([ self.points_per_network, len(added_X)])) ], self.max_iterations, self.sp) ) )
            if self.parallelize == False:
              self.networks.append( ( mn, mx, NeuralLinearNetwork( self.W0 , self.num_layers, self.step, self.max_iterations, self.sp) ) )
            else:
              self.networks.append( ( mn, mx, createNetworkForNode(('', 50000), 'abc', self.W0, self.num_layers, self.step, self.max_iterations) ) )
#             _added_X = [ [ x[0] for x in added_X ] ]
#             _added_Y = [ [ y[0] for y in added_Y ] ]
            if self.parallelize == False:
              self.networks[len(self.networks)-1][2].study(added_X, added_Y)
            else:
              if self.pool == None:
                self.pool = TinyThreadPool(10)
                self.pool.start()
#              print "enqueue"
              self.pool.enqueue_task_id((len(self.networks)-1)%10,NeuralLinearComposedNetwork.nstudy_wrapper,self, self.networks[len(self.networks)-1][2],added_X, added_Y)

            for j in xrange(0, len(self.W0)):
              if mn[j] < self.mn[j][0] or self.mn[j][0] == None:
                self.mn[j] = ( mn[j], len(self.networks)-1 )
              if mx[j] > self.mx[j][0] or self.mx[j][0] == None:
                self.mx[j] = ( mx[j], len(self.networks)-1 )
#            print "Added mn:", mn, " mx:", mx,"added_X:", added_X,"added_Y:", added_Y, "self.mn:",self.mn,"self.mx:", self.mx
          added_X = [ ]
          added_Y = [ ]
          if self.pool != None:
            self.pool.wait_ready()

    def getWeights(self, mn, mx, j, layer):
      for network in self.networks:
        if network[0] <= mn and network[1] >= mx:
          return network[2].getWeights(layer)

    def calc_y2(self, X, Y, up_to = None):
      network = None
      i = None
      num_match = [ 0 for j in self.networks ]
      distance = [ [ 0, 0 ] for j in self.networks ]
      found = False
      XX = copy.deepcopy(X)
      for j in xrange(0, len(self.W0)):
#        print "looking at arg ", j
        _X = [ ]
#        print self.cyclic
        if self.cyclic[j] == True:
#          print "CYCLIC was detected by neural network"
#          print "min arg:", self.mn, "max arg:", self.mx, "X:", X, "arg num:", j
          _X.append(self.mn[j][0] + X[j] % ( self.mx[j][0] - self.mn[j][0] + 1))
          XX[j] = self.mn[j][0] + X[j] % ( self.mx[j][0] - self.mn[j][0] + 1)
#          print "_X:",_X, self.cyclic[j]
        else:
          _X.append(X[j])
#          print "_X:",_X
        for n in xrange(0, len(self.networks)):
#          print "looking at network candidate ", n, self.networks[n], _X[0]
#          print "mmin:", self.networks[n][0][j], "max:", self.networks[n][1][j],"X[j]:", _X[0]
          if (self.networks[n][0][j] <= _X[0] and self.networks[n][1][j] >= _X[0]):
#            print "Matches"
            network = self.networks[n]
            i = n
            distance[n][0] = abs(self.networks[n][0][j] - _X[0] )
            distance[n][1] = abs(self.networks[n][1][j] - _X[0] )
#            print "Calc distance: ", distance[j]
            num_match[n] += 1
#            print "Network ", n, "num matches ", num_match[n]
#            print self.networks[n][0][j], self.networks[n][1][j], num_match[n]
          else:
            distance[n][0] = abs(self.networks[n][0][j] - _X[0] )
            distance[n][1] = abs(self.networks[n][1][j] - _X[0] )
#            print "Calc distance: ", distance[j]
#      print self.networks
      if i != None:
#        print "chosen2:",self.networks[num_match.index(max(num_match))], max(num_match), num_match, num_match.index(max(num_match))
        YY = self.networks[num_match.index(max(num_match))][2].calc_y2(XX, Y, up_to)
        for y in  xrange(0, len(Y)):
          Y[y] = round(copy.deepcopy(YY[y]),2)
      else:
#        print distance
        dist = [ d[0] + d[1] for d in distance ]
#        print "chosen3:", dist, dist.index(min(dist))
        YY = self.networks[dist.index(min(dist))][2].calc_y2(XX, Y, up_to)
        for y in  xrange(0, len(Y)):
          Y[y] = round(copy.deepcopy(YY[y]),2)
      return XX


def convert_array_str(s):
    arr = [ ]
    acc = ""
    for a in s[:-1]:
      acc += a
      print acc
      if len(acc) >= 1:
        if acc != '':
          arr.append(int(acc))
        acc = ""
    if acc != '':
      arr.append(int(acc))
    return arr


def getMean(mvec):
#    print mvec
    s = [ 0 for i in xrange(0, len(mvec[0])) ]
    for m in xrange(0, len(mvec)):
      for j in xrange(0, len(mvec[m])):
        s[j] += mvec[m][j]
    mid = [ float(s[i])/float(len(mvec)) for i in xrange(0, len(s)) ]
    return mid

def getVecDelta(vec1, vec2):
    vec = [ 0 for i in xrange(0, len(vec1)) ]
    for v in xrange(0, len(vec1)):
      vec[v] = abs(vec1[v]-vec2[v])
    return vec

def getAverageDelta(vec, mean):
    s = 0.0
    for v in vec:
      s+= getVecDiffAbs(v, mean)
    s = s / float(len(vec))
    return s

def getVecAverageDelta(vec, mean):
    s = [ 0.0 for i in xrange(0,len(vec[0])) ]
    for v in vec:
      d = getVecDelta(v, mean)
      for j in xrange(0, len(vec[0])):
        s[j] += d[j]
    for j in xrange(0, len(vec[0])):
      s[j] = s[j] / float(len(vec))
    return s

class Cluster:
    def __init__(self, vec = [ ], parent = None, name = None):
      self.vec = copy.deepcopy(vec)
      self.mean = None
      self.av_delta = None
      self._recalc()
      self.parent = parent
      self.subclusters = [ ]
      self.name = name
      self.err = 0.1
    def _recalc(self):
      if len(self.vec) > 0:
        self.mean = getMean(self.vec)
        self.av_delta = getVecAverageDelta(self.vec, self.mean)
        return True
      return False
    def check_delta(self, vec):
      d1 = getVecDelta(vec, self.mean)
      for j in xrange(0, len(d1)):
        if (d1[j] - self.av_delta[j]) >= self.err:
          return False
      return True
    def classify(self, vec):
#      print self.mean
#      print "Average ", self.av_delta
      if not self._recalc():
        self.vec.append(vec)
        self._recalc()
        return True
      left = True
      if not self.check_delta(vec):
          left = False
      if left == True:
        self.vec.append(vec)
        self._recalc()
        return True
      return False
    def clusterize(self):
        if not self._recalc():
          return [ ]
#        print self.mean, self.av_delta
        vec1 = [ ]
        vec2 = [ ]
        for v in self.vec:
          left = True
          if not self.check_delta(v):
              left = False
          if left == True:
            vec1.append(v)
          else:
            vec2.append(v)
        if len(vec1)>0 and len(vec1)!=len(self.vec):
          c1 = Cluster(vec1, self)
          self.subclusters.append(c1)
        if len(vec2)>0 and len(vec2)!=len(self.vec):
          c2 = Cluster(vec2, self)
          self.subclusters.append(c2)
        return self.subclusters

    def k_means(c, num_splits=None):
#        print "k_means: ", c, c.vec, c.parent
        k = [ ]
        if num_splits == None:
          num_splits = len(c.vec)
        else:
          num_splits = min([len(c.vec), num_splits])
        if len(c.vec) > 1 and num_splits > 0:
          k.append(c)
          res = c.clusterize()
          for r in res:
            if len(r.vec) > 0:
              r1 = Cluster.k_means(r, num_splits-1)
              k.extend(r1)
        else:
          if len(c.vec) > 0:
            k.append(c)
        return k

class Classificator:
    def __init__(self, init_vec = None):
      self.init_vec = [ ]
      self.clusters = [ ]
      self.cluster = None
      if init_vec != None:
        self.reinit(init_vec)
    def add_cluster(self, cluster):
      self.init_vec.extend(cluster.vec)
      if self.cluster == None:
        self.cluster = Cluster(self.init_vec)
      self.cluster.subclusters.append(cluster)
      self.clusters.append(cluster)
    def reinit(self, init_vec, add = True):
      if add == True:
        self.init_vec.extend(init_vec)
      else:
        self.init_vec = copy.deepcopy(init_vec)
#      print self.init_vec
      self.cluster = Cluster(self.init_vec)
      self.clusters  = Cluster.k_means(self.cluster)
#      print self.clusters
    def classify_vec(self, vec, first_or_smallest = False, only_first = True):
      cluster_map = { }
      self.print_info()
      for v in xrange(0, len(vec)):
        c = self.classify(vec[v], first_or_smallest)
        print "c:",c, vec[v]
        if c!=None:
          if not c in cluster_map:
            cluster_map[c] = 1
          else:
            cluster_map[c] += 1
      print cluster_map
      cluster_vec = [ (c,cluster_map[c]) for c in cluster_map ]
#      print cluster_vec
      clusters = [ ]
      for c in cluster_vec:
          if c[1] > 0:
            clusters.append(c)
      clusters.sort()
      print clusters
      if only_first == True:
        if len(clusters)>0:
          return clusters[0][0]
        else:
          return None
      return clusters
    def classify(self, vec, first_or_smallest = False):
      print "Classiffy: ", self.clusters
      smallest = None
      if first_or_smallest == True:
        for c in self.clusters:
          if c.classify(vec):
            return c
      else:
        for c in self.clusters:
          print "Classify ", c, vec
          if c.classify(vec):
            if smallest != None:
              if len(c.vec) < len(smallest.vec):
                smallest = c
            else:
                smallest = c
        return smallest
      return None
    def print_info(self):
      for c in self.clusters:
        print "Cluster ", c, "Parent: ", c.parent, " Vec: ", c.vec, "Mean: ", c.mean, "Av delta: ", c.av_delta
        print "Subclusters: ", c.subclusters


class Predictor:
    def __init__(self, points_per_network, W, layers, step, max_steps, acc_value = 1):
      self.neural = NeuralLinearComposedNetwork(points_per_network, W, layers, step, max_steps);
      self.analyzer = FactorAnalyzer()
      self.W = W
      self.acc_value = acc_value
      self.acc_x = [ ]
      self.acc_y = [ ]
      self.classificator = Classificator()
    def study(self, X, Y):
      for i in xrange(0, len(X)):
        self.acc_x.append(X[i])
        self.acc_y.append(Y[i])
      if len(self.acc_x) < self.acc_value:
        return
      acc = [ ]
      for x in xrange(0, len(self.acc_x)):
        a = copy.deepcopy(self.acc_x[x])
        a.extend(self.acc_y[x])
        acc.append(a)
      self.classificator.reinit(acc)
      self.neural.study(self.acc_x, self.acc_y)
      for x in xrange(0, len(self.acc_x)):
        self.analyzer.analyze_factor(str(self.acc_x[x]), generate_entity_x, str(self.acc_y[x]))
      self.acc_x = [ ]
      self.acc_y = [ ]
    def all_and(self, arr):
      res = True
      for a in arr:
        res = res and a
      return res
    def predict_p(self, prefix, Y, P, depth, is_prefix_time = None):
      classes = [ ]
      if is_prefix_time == None:
        is_prefix_time = False if self.all_and(self.neural.cyclic) == True else True
      prefix_vector = [ str(e)  for e in prefix ]
      if is_prefix_time == False:
#        print "Periodic"
        res = self.analyzer.deduct(prefix_vector, depth, generate_entity_x )
        for p in xrange(0, len(prefix_vector)):
          prefix[p] = eval(prefix_vector[p])
        if depth > len(res):
          res = None
#        print res
        for p in prefix:
          Yout = [ 0 for i in xrange(0, len(self.W)) ]
          appr_p = self.neural.calc_y2( p, Yout)
          Y.append(Yout)
          P.append(p)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
        if res == None:
#          print xrange(int(prefix[0][0]), int(prefix[0][0]+depth))
          for x in xrange(int(prefix[0][0]), int(prefix[0][0]+depth)):
            Yout = [ 0 for i in xrange(0, len(self.W)) ]
            appr_p = self.neural.calc_y2([ x for i in xrange(0, len(self.W)) ], Yout)
#            print "Yout:", Yout
            Y.append(Yout)
            P.append([ x for i in xrange(0, len(self.W)) ])
            if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
          return classes
        for r in res:
          Yout = [ 0 for i in xrange(0, len(self.W)) ]
#          print r.data
          P.append(eval(r.data))
          appr_p = self.neural.calc_y2(eval(r.data), Yout)
#          print "Yout:", Yout
          Y.append(Yout)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
      else:
#        print "HERE:"
        for p in prefix:
          Yout = [ 0 for i in xrange(0, len(self.W)) ]
          appr_p = self.neural.calc_y2( p, Yout)
          Y.append(Yout)
          P.append(p)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
        X = [ ]
        r = [ ]
        _prefix = prefix[len(prefix)-1]
        XVec = copy.deepcopy(_prefix)
        for x in xrange(0, depth):
          for j in xrange(0, len(_prefix)):
            XVec[j]+=1
          X.append(copy.deepcopy(XVec))
#        print "HERE:", X
        for x in X:
          Yout = [ 0 for i in xrange(0, len(self.W)) ]
#          Yout = [ ]
          appr_p = self.neural.calc_y2( x, Yout)
          Y.append(Yout)
          P.append(x)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
      return classes

    def predict_p_classes(self, prefix, Y, P, depth, classes, is_prefix_time = None):
      print "Classes: ", classes
      self.classificator.print_info()
      if is_prefix_time == None:
        is_prefix_time = False if self.all_and(self.neural.cyclic) == True else True
      prefix_vector = [ str(e)  for e in prefix ]
      if is_prefix_time == False:
#        print "Periodic"
        res = self.analyzer.deduct(prefix_vector, depth, generate_entity_x )
        for p in xrange(0, len(prefix_vector)):
          prefix[p] = eval(prefix_vector[p])
        if depth > len(res):
          res = None
#        print res
        for p in prefix:
          Yout = [ 0 for i in xrange(0, len(self.W)) ]
          appr_p = self.neural.calc_y2( p, Yout)
          Y.append(Yout)
          P.append(p)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
        if res == None:
#          print xrange(int(prefix[0][0]), int(prefix[0][0]+depth))
          for x in xrange(int(prefix[0][0]), int(prefix[0][0]+depth)):
            Yout = [ 0 for i in xrange(0, len(self.W)) ]
            appr_p = self.neural.calc_y2([ x for i in xrange(0, len(self.W)) ], Yout)
#            print "Yout:", Yout
            Y.append(Yout)
            P.append([ x for i in xrange(0, len(self.W)) ])
            if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
          return
        for r in res:
          Yout = [ 0 for i in xrange(0, len(self.W)) ]
#          print r.data
          P.append(eval(r.data))
          appr_p = self.neural.calc_y2(eval(r.data), Yout)
#          print "Yout:", Yout
          Y.append(Yout)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
      else:
#        print "HERE:"
        for p in prefix:
          Yout = [ 0 for i in xrange(0, len(self.W)) ]
          appr_p = self.neural.calc_y2( p, Yout)
          Y.append(Yout)
          P.append(p)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))
        X = [ ]
        r = [ ]
        _prefix = prefix[len(prefix)-1]
        XVec = copy.deepcopy(_prefix)
        for x in xrange(0, depth):
          for j in xrange(0, len(_prefix)):
            XVec[j]+=1
          X.append(copy.deepcopy(XVec))
#        print "HERE:", X
        for x in X:
          Yout = [ 0 for i in xrange(0, len(self.W)) ]
#          Yout = [ ]
          appr_p = self.neural.calc_y2( x, Yout)
          Y.append(Yout)
          P.append(x)
          if classes != None:
              acc = copy.deepcopy(appr_p)
              acc.extend(Yout)
              classes.append(self.classificator.classify(acc))

def autoCorrelation( Y, n ):
  Corr = 0
  for m in xrange(0, len(Y)):
    Corr+=abs((Y[m][0] - Y[(m+n) % len(Y)][0]))
  return Corr


def detectPeriodic(  Y ):
  prev_corr = None
  Corr = None
  min_corr = None 
  num_periods = 0
  df = 0
  prev_df = 0
  for i in xrange(0, len(Y)/2):
#    print i
    prev_corr = Corr
    Corr = autoCorrelation( Y, i )
#    print "Corr:",Corr
    if prev_corr != None:
      prev_df = df
      df = abs(prev_corr - Corr)
#      print df
      if prev_corr > Corr or prev_df > df:
        num_periods += 1
        if min_corr == None or min_corr[0] > Corr:
          min_corr = ( Corr, i )
    else:
      prev_corr = Corr
#      print Corr
  if min_corr != None and num_periods >= 1:
#    print "Periodic detected maximum correlation at ", min_corr[1], Y[min_corr[1]]
    return True
  return False

def square_func(X, n):
  Y = [ [ sum(x)*sum(x) for j in xrange( 0, n ) ] for x in X ]
  return Y

def linearTest():
    print "Linear(extrapolate) test begin"
    W = [ 1.0, 1.0 ]
    x0 = [ 1.0, 2.0 ]
    y0 = [ 9.0, 9.0 ]
    x1 = [ 5.0, 6.0 ]
    y1 = [ 121.0, 121.0 ]
    x2 = [ 9.0, 10.0 ]
    y2 = [ 19.0*19.0, 19.0*19.0 ]
    x3 = [ 13.0, 14.0 ]
    y3 = [ 27.0*27.0, 27.0*27.0 ]
    X = [ x0, x1, x2, x3 ]
    Y = [ y0, y1, y2, y3 ]
    X.append([ 20.0, 20.0 ])
    Y.append([ 1600.0, 1600.0 ])
#    Y.append([ 36.0, 36.0 ])
    P = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    p = Predictor(2, Wout, 3, step, 1000000)
    p.study(X, Y)
    Yout = [ ]
    p.predict_p([ [ 13.0, 14.0 ] ], Yout, P, 10)
    print "Approximated Y:", Yout
    print "Approximated X:", P
    print "Y:", Y
    print "X:", X
    print "Linear test end"
    p.neural.pool.wait_ready()
    p.neural.pool.stop()

def periodicTest():
    P = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    W = [ 1.0, 1.0 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    print "Periodic(sin emulate) test begin"
    p2 = Predictor(2, Wout, 5, step, 1000000)
    Y = [ [math.sin(i), math.sin(i)] for i in xrange(0,10) ]
    X = [ [ i, i ] for i in xrange(0,10) ]
    print Y, [ [i, i] for i in xrange(0,10) ]
    p2.study([ [i, i] for i in xrange(0,10) ], Y)
    Yout = [ ]
    P = [ ]
    _classes = p2.predict_p([ [ 0, 0 ] ], Yout, P, 10)
    for c in xrange(0, len(_classes)):
      if _classes[c] != None:
        print "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec
      else:
        print "P: ", P[c], Yout[c], "Class: None"
    print "Approximated Sin:", Yout
    print "Approximated X:", P
    print "Sin(X): ", Y
    print "X: ", X
    print "####"
    print "Periodic(sin) test end"
    p2.neural.pool.wait_ready()
    p2.neural.pool.stop()

def periodicRandTest():
    P = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    W = [ 1.0, 1.0 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    print "Periodic(sin emulate) test begin"
    p2 = Predictor(1, Wout, 5, step, 1000000)
    Y = [ [math.sin(i)+random.randint(0,100), math.sin(i)+random.randint(0,100)] for i in xrange(0,10) ]
    X = [ [ i, i ] for i in xrange(0,10) ]
    print Y, [ [i, i] for i in xrange(0,10) ]
    p2.study([ [i, i] for i in xrange(0,10) ], Y)
    Yout = [ ]
    P = [ ]
    p2.predict_p([ [ 0, 0 ] ], Yout, P, 20)
    print "Approximated Sin+rand:", Yout
    print "Approximated X:", P
    print "Sin(X)+rand: ", Y
    print "X: ", X
    print "####"
    print "Periodic(sin) test end"
    p2.neural.pool.wait_ready()
    p2.neural.pool.stop()

def logicTest():
    P = [ ]
    Y = [ ]
    X = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    W = [ 1.0, 1.0 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1] ]
    step = [ 0.01, 0.01 ]
    _classes = [ ]
    print "Logic test begin"
    p3 = Predictor(2, Wout, 3, step, 1000000)
    p3.study( [ [ 1.0, 0.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 1.0, 0.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 1.0 ] ], [ [ 2.0, 2.0 ] ] )
    X.append( [ 1.0, 1.0 ] )
    Y.append( [ 2.0, 2.0 ] )
    p3.study( [ [ 1.0, 1.0 ] ], [ [ 2.0, 2.0 ] ] )
    X.append( [ 1.0, 1.0 ] )
    Y.append( [ 2.0, 2.0 ] )
    p3.study( [ [ 0.0, 0.0 ] ], [ [ 0.0, 0.0 ] ] )
    X.append( [ 0.0, 0.0 ] )
    Y.append( [ 0.0, 0.0 ] )
    p3.study( [ [ 0.0, 1.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 0.0, 1.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 0.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 1.0, 0.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 0.0, 1.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 0.0, 1.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 0.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 1.0, 0.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 1.0 ] ], [ [ 2.0, 2.0 ] ] )
    X.append( [ 1.0, 1.0 ] )
    Y.append( [ 2.0, 2.0 ] )
    p3.study( [ [ 1.0, 1.0 ] ], [ [ 2.0, 2.0 ] ] )
    X.append( [ 1.0, 1.0 ] )
    Y.append( [ 2.0, 2.0 ] )
    p3.study( [ [ 0.0, 0.0 ] ], [ [ 0.0, 0.0 ] ] )
    X.append( [ 0.0, 0.0 ] )
    Y.append( [ 0.0, 0.0 ] )
    p3.study( [ [ 0.0, 1.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 0.0, 1.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    p3.study( [ [ 1.0, 0.0 ] ], [ [ 1.0, 1.0 ] ] )
    X.append( [ 1.0, 0.0 ] )
    Y.append( [ 1.0, 1.0 ] )
    Yout = [ ]
    P = [ ]
    p3.predict_p_classes([ [ 1.0, 3.0 ] ], Yout, P, 10, _classes)
    print "Approximated Y: ", Yout
    print "Approximated X: ", P
    print "Y:", Y
    print "X:", X
    print "Logic test end"
    for c in xrange(0, len(_classes)):
      if _classes[c] != None:
        print "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec
      else:
        print "P: ", P[c], Yout[c], "Class: None"
    p3.neural.pool.wait_ready()
    p3.neural.pool.stop()

def logicTest2():
    P = [ ]
    Y = [ ]
    X = [ ]
#    X = [ y0, y1, y2 ]
#    Y = [ x0, x1, x2 ]
    W = [ 1.0, 1.0 ]
    grad = [ 0, 0 ]
    Wout = [ W[0], W[1],W[0], W[1], W[0], W[1],W[0], W[1] ]
    step = [ 0.01, 0.01,0.01, 0.01,0.01, 0.01,0.01, 0.01 ]
    _classes = [ ]
    print "Logic test begin"
    p3 = Predictor(1, Wout, 3, step, 1000000)
    print "Predictor created"
    X.append([ 0, 1, 1, 0, 0, 1, 1, 0])
    Y.append([ 1, 0, 0, 0, 0, 0, 0, 0])
    X.append([ 1, 0, 0, 1, 1, 0, 0, 1])
    Y.append([ 0, 0, 0, 0, 0, 0, 0, 1])
    X.append([ 1, 0, 1, 1, 1, 1, 0, 1])
    Y.append([ 0, 1, 1, 1, 1, 1, 1, 0])
    print "Before study"
    p3.study(X, Y)
    print "After study"
    return
    Yout = [ ]
    P = [ ]
    p3.classificator.print_info()
    p3.predict_p_classes([ [ 1, 1, 1, 0, 0, 1, 1, 0 ] ], Yout, P, 1, _classes, False)
    print "Approximated Y: ", Yout
    print "Approximated X: ", P
    print "Y:", Y
    print "X:", X
    print "Logic test end"
    for c in xrange(0, len(_classes)):
      if _classes[c] != None:
        print "P: ", P[c], Yout[c], "Class: ", _classes[c], _classes[c].vec
      else:
        print "P: ", P[c], Yout[c], "Class: None"
#    p3.neural.pool.wait_ready()
#    p3.neural.pool.stop()


def study_thread(p, f):
    i = 0
    X = [ ]
    Y = [ ]
    while(True):
      print "before readline"
      line = f.readline()
      print "readline ", line
      s = "[ " + line + " ] "
      v = eval(s)
      if (i & 1) == 0:
        X.append(v)
        print X, i
      else:
        Y.append(v)
        print "Study:"
        print "X: "
        print X
        print "Y: "
        print Y
        p.study(X, Y)
      i+=1

def classifierTest():
    print "Classifier test begin"
    circle = [ ]
    a = 0
    while a < 360:
      x = 10*math.cos(a)
      y = 10*math.sin(a)
      circle.append([x,y])
      a+=30

    square = [ ]
    square.append([0,0])
    square.append([0,5])
    square.append([5,5])
    square.append([5,0])
    c1 = Cluster(square, None, "square")
    print circle
    c2 = Cluster(circle, None, "circle")
    print circle
    c = Classificator()
    c.add_cluster(c1)
    c.add_cluster(c2)
    print circle
    res = c.classify_vec(square)
    if res:
      print "Found:", res, res.name
    else:
      print "Not found"
    print square
    res = c.classify_vec(circle)
    if res:
      print "Found:", res, res.name
    else:
      print "Not found"
    print circle
    circle.extend(square)
    res = c.classify_vec(circle, False, False)
    for r in res:
      print "Found:", r[0], r[0].name
    c.print_info()
    print "Classifier test end"

def classifierTest2():
    print "Classifier2 test begin"
    c = Classificator()
    X = [ ]
    Y = [ ]
    X.append([ 0, 1, 1, 0, 0, 1, 1, 0])
    Y.append([ 1, 0, 0, 0, 0, 0, 0, 0])
    X[0].extend(Y[0])
    c.reinit(X)
    X.append([ 1, 0, 0, 1, 1, 0, 0, 1])
    Y.append([ 0, 0, 0, 0, 0, 0, 0, 1])
    X[1].extend(Y[1])
    c.reinit(X)
    c.print_info()
    print "Classifier2 test end"

def predict_thread(p, f, f2):
    i = 0
    while(True):
      X = [ ]
      Y = [ ]
      P = [ ]
      line = f.readline()
      depth = int(f.readline())
      s = "[ " + line + " ] "
      v = eval(s)
      X.append(v)
      print "Predict:"
      print "X: "
      print X
      print "Y: "
      print Y
      print "P: "
      print P
      print "depth:"
      print depth
      p.predict_p(X, Y, P, depth)
      f2.write(str(P)+"\n")
      f2.write(str(Y)+"\n")
      f2.flush()

def run_all_tests():
    oldstdout = sys.stdout
#    s = "dsdsdsds"
#    try:
    mystdout = StringIO()
    sys.stdout = mystdout
#    print "sdsds"
#    linearTest()
#    periodicTest()
#    periodicRandTest()
#    logicTest()
#    classifierTest()
    logicTest2()
#    classifierTest2()
    s = mystdout.getvalue()
#    except e:
#      print "Exception!"
#    finally:
    sys.stdout = oldstdout
    return s

if __name__ == "__main__":
#    Y = [ [1.0 ], [ 1.0 ], [ 2.0 ],  [ 2.0 ], [ 0.0 ], [ 1.0 ], [1.0] ]
#    Y = [ [ 1.0 ], [ 2.0 ],  [ 2.0 ]]
#    print detectPeriodic(Y)
#    exit(0)
#    linearTest(7, 2, 0.01, square_func)
#    exit(0)
#    vec = [[-1,0],[0,1], [3, 0], [ 10, 10 ]]
    if len(sys.argv) > 1:
      if sys.argv[1] == "create_network":
        l = 0
        W = [ ]
        for i in xrange(0, int(sys.argv[3])):
          W.append(float(sys.argv[4+i]))
        l+=4+int(sys.argv[3])
        l1 = l
        l+=1
        step = [ ]
        for i in xrange(0, len(W)):
          step.append(float(sys.argv[l+i]))
        l += len(W)
        p = Predictor(int(sys.argv[2]), W, int(sys.argv[l1]), step, int(sys.argv[l]))
        print "Network created: Points per network:", int(sys.argv[2]), "W: ", W, "layers: ", int(sys.argv[l1]), "step: ", step, \
        "max_iterations: ", int(sys.argv[l])
        input_file = sys.argv[l+1]
        predict_file = sys.argv[l+2]
        output_file = sys.argv[l+3]
        f = open(input_file, "r+b")
        f2 = open(predict_file, "w+b")
        f3 = open(output_file, "w+b")
        t1 = Thread(target = study_thread, args = ( p, f ) )
        t2 = Thread(target = predict_thread, args = ( p, f2, f3 ) )
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        p.neural.pool.wait_ready()
        p.neural.pool.stop()
      if sys.argv[1] == "server":
        print "Started as a server"
        manager = NeuralNetworkManager(address=('', 50000), authkey='abc')
        server = manager.get_server()
        server.serve_forever()
      if sys.argv[1] == "client":
        print "Started as a client ", os.getpid()
        network = createNetworkForNode(('', 50000), 'abc', [ 1.0, 1.0 ], 2, [ 0.1, 0.1 ], 1000000)
        network.study( [ [2.0, 2.0] ], [ [4.0, 4.0 ] ])
        Yout =  [ 0.0, 0.0 ]
        Yout = network.calc_y2([2.2, 2.2], Yout )
        print Yout
      if sys.argv[1] == "server2":
        pid = os.spawnlp(os.P_NOWAIT, "ssh", "ssh", "localhost", 'cd /home/estalis/exps/outcome2/;python predict.py server &')
        print pid
#        commands.getoutput("ssh localhost 'cd /home/estalis/exps/outcome2/;python predict.py server &'")
      exit(0)
    linearTest()
    periodicTest()
    periodicRandTest()
    logicTest()
    classifierTest()
    logicTest2()
    classifierTest2()