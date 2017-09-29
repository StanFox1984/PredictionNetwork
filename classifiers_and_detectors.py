import random
import copy
import math
import time
from difflib import SequenceMatcher


#some amount of data is constantly accumulated
#values ranges is calculated and for each range
#the best matching classifer is calculated(summary error)
#ranges are calculated by initially auto classifying
#data
#then for each input its range is calculated and then
#correspondent classifer is applied
#the system can also run on its own, in that case
#each next input is obtained from the shadow prob_classifer or polynom classifer in
#the given range, for prob classifer next state is based on probability, while
#for polynom its based on gradient vector direction of the accumulated data

def sign(X, Y):
    res = [ ]
    for x,y in zip(X,Y):
        if x != y:
           res.append((x-y)/abs(x-y))
        else:
           res.append(0)

    return res

class ClassifierDetector:

    def CreateClassifier(self, state):
        c = ProbClassifier()
        X = [ ]
        Y = [ ]
        state_for_prob = [ ]
        for s in state:
            l = int(math.ceil(float(len(s))/2.0))
            X.append(s[:l])
            Y.append(s[l:])

        even = True
        i = 0
        for j in xrange(0, 2*len(state)):
            if even == True:
                state_for_prob.append(X[i])
                even = False
                print "Added X ", X[i]," to state_for_prob from ", X
            else:
                state_for_prob.append(Y[i])
                even = True
                print "Added Y ", Y[i]," to state_for_prob from", Y
                i += 1

        if self._is_polynom(X, Y):
            print X, Y, " classified as PolynomClassifier"
            c = PolynomClassifierChain(3, len(X[0]))

        if self._is_detect(X, Y):
            print X, Y, " classified as DetectClassifier"
            c = DetectClassifier()

        if self._is_prob(X, Y):
            print X, Y, " classified as ProbClassifier"
            c = ProbClassifier()

        if not isinstance(c, ProbClassifier):
            for s in state:
                c.new_state(s)
        else:
            for s in state:
                print "ProbClassifier new state ", s
                c.new_state(s)

        return c


    def _is_prob(self, input_data, output_data):
        if self._is_polynom(input_data, output_data):
            return False
        if self._is_detect(input_data, output_data):
            return False
        print "Function is not polynom or detect - choosing Probabilistic as default"
        return True

    def _is_monotonic(self, input_data, output_data):
        last_sign_X = None
        last_sign_Y = None
        for i in xrange(0, len(input_data)):
            if i > 0:
                if last_sign_X == None or last_sign_Y == None:
                    last_sign_X = sign(input_data[i], input_data[i-1])
                    last_sign_Y = sign(output_data[i], output_data[i-1])
                else:
                    sign_X = sign(input_data[i], input_data[i-1])
                    sign_Y = sign(output_data[i], output_data[i-1])
#                    print "sign: ", sign_X, sign_Y
                    ratio1 = [ abs(last_sign_x) - abs(sign_x) for last_sign_x, sign_x in zip(last_sign_X, sign_X) ]
                    ratio2 = [ abs(last_sign_y) - abs(sign_y) for last_sign_y, sign_y in zip(last_sign_Y, sign_Y) ]

                    print ratio1, ratio2
                    if ratio1 != ratio2:
                        print "Function is not monotonic"
                        return False

                    if last_sign_X != sign_X:
                        print "X is not monotonic"
                        return False

                    last_sign_X = sign_X
                    last_sign_Y = sign_Y
#                print "last_sign: ", last_sign_X, last_sign_Y
        print "Function is monotonic"
        return True

    def _is_polynom(self, input_data, output_data):
        if not self._is_monotonic(input_data, output_data):
            return False
        if not self._is_bijective(input_data, output_data):
            print "Not bijective => not polynom"
            return False
        print "Function is monotonic"
        return True

    def _is_bijective(self, input_data, output_data):
        output_data_map = { }
        for i in xrange(0, len(input_data)):
            if str(input_data[i]) not in output_data_map:
                output_data_map[str(input_data[i])] = output_data[i]
            else:
                if output_data[i] != output_data_map[str(input_data[i])]:
                    print "Same input data correspond to different output - not bijective"
                    return False
        print "Bijective reflection is detected"
        return True

    def _is_detect(self, input_data, output_data):
        if self._is_monotonic(input_data, output_data):
            return False
        if not self._is_bijective(input_data, output_data):
            return False
        print "Bijective reflection is detected - could be Detector"
        return True

def testClassifierDetector():
    input_data = [ [-2], [0],[1],[2] ]
    output_data = [ [-4], [0],[2],[4] ]

    c = ClassifierDetector()
    if c._is_polynom(input_data, output_data) == False:
        print "Polynom test not passed10"
        return False

    if c._is_detect(input_data, output_data) == True:
        print "Detect test not passed10"
        return False

    if c._is_prob(input_data, output_data) == True:
        print "Prob test not passed10"
        return False

    input_data = [ [-2], [0],[1],[2] ]
    output_data = [ [4], [0],[1],[4] ]

    c = ClassifierDetector()
    if c._is_polynom(input_data, output_data) == False:
        print "Polynom test not passed11"
        return False

    if c._is_detect(input_data, output_data) == True:
        print "Detect test not passed11"
        return False

    if c._is_prob(input_data, output_data) == True:
        print "Prob test not passed11"
        return False


    print "Polynom input passed"

    input_data = [ [0],[2],[1],[2] ]
    output_data = [ [0],[2],[4],[3] ]

    if c._is_polynom(input_data, output_data) == True:
        print "Polynom test not passed2"
        return False

    if c._is_detect(input_data, output_data) == True:
        print "Detect test not passed2"
        return False

    if c._is_prob(input_data, output_data) == False:
        print "Prob test not passed2"
        return False

    print "Prob input passed"

    input_data = [ [0, 0],[2,2],[1,1],[2,2] ]
    output_data = [ [0],[2],[4],[2] ]

    if c._is_polynom(input_data, output_data) == True:
        print "Polynom test not passed3"
        return False

    if c._is_detect(input_data, output_data) == False:
        print "Detect test not passed3"
        return False

    if c._is_prob(input_data, output_data) == True:
        print "Prob test not passed3"
        return False

    print "Detect input passed"

    print "All tests passed for ClassifierDetector"

    return True

class ClassifierApplier:
    def __init__(self, strict_output = True):
        self.classifiers = { }
        self.states_map = { }
        self.data = { }
        self.c = DetectAutoClassifier()
        self.det = ClassifierDetector()
        self.data_stamps = { }
        self.data_stamp = 0
        self.prob_classifier = ProbClassifier()
        self.strict_output = strict_output

    def new_state(self, state):
        self.prob_classifier.new_state(state)
        self.c.new_state(state)
        self.data_stamps[str(state)] = self.data_stamp
        self.data_stamp += 1
        arr = [ ]
        self.states_map = { }
        self.data = { }

        print "Classifier applier: ", self.c.classes
        for c in self.c.classes:
            if c[1] not in self.data:
                print "class ", c[1], " was not in data: ", self.data
                self.data[c[1]] = [ ]
            print "Adding data ", c[0], " to self.data[",c[1],"]: ", self.data
            self.data[c[1]].append(c[0])
            self.states_map[str(c[0])] = c[1]

        for c in self.c.classes:
            classifier_data = sorted([ [ self.data_stamps[str(state)], state ] for state in self.data[c[1]] ])
            self.classifiers[c[1]] = self.det.CreateClassifier([ d[1] for d in classifier_data ])

    def apply(self, _input_value = None):
        if _input_value == None:
            _input_value = self.prob_classifier.apply()
        input_value = None
        class_num = None
        if str(_input_value) not in self.states_map:
            similar =  sorted([ [ ProbClassifier._similarity(str(_input_value), s), s ] for s in self.states_map ])
            similar.reverse()
            print similar
            input_value = similar[0][1]
        else:
            input_value = str(_input_value)

        class_num = self.c.apply(eval(input_value))

#        print "input ", str(_input_value)," is of class ", self.states_map[input_value]
        print "input ", str(input_value)," is of class ", class_num
        print self.data
        print "used for data: ", self.data[class_num]
        return self.classifiers[class_num].apply(eval(input_value))
        #for each data range the best matching classifer type is applied


class ProbNode:
    def __init__(self, name):
        self.name = name
        self.incomes = { }
        self.outcomes = { }
    def next_hit(self, node):
        node_prob = None
        if node.name not in self.outcomes:
            node_prob = [ node, 0.0, 0 ]
            self.outcomes[node.name] = node_prob
            print "Added node ", node.name, " to ", self.name
        else:
            node_prob = self.outcomes[node.name]
        node_prob[2] += 1.0
        if self.name not in node.incomes:
            node.incomes[self.name] = [ self, node_prob[1], node_prob[2] ]
        self._recalc_probs()
        print "Now node ", self.name, " outcomes are: ", self.outcomes
    def _recalc_probs(self):
        print self.name, "->", self.outcomes
        hits_sum = sum([ self.outcomes[name][2] for name in self.outcomes ])
        for name in self.outcomes:
            node_prob = self.outcomes[name]
            node = self.outcomes[name][0]
            node_prob[1] = node_prob[2]/hits_sum
            node.incomes[self.name][1] = node_prob[1]
            print node.name, "->", node.outcomes

    def next_simulate(self):
        rval = random.random()*100.0
        ranges = [ ]
        last_val = 0
        node = None
        print "next_simulate node ",self.name," outcomes: ", self.outcomes
        for k in self.outcomes:
            n = self.outcomes[k]
            delta = 100.0 * n[1]
            ranges.append([ [last_val, last_val + delta], n[0] ])
            last_val += delta
        print "ranges:", ranges
        if len(ranges) == 0:
            return self
        for r in ranges:
            if rval >= r[0][0] and rval <= r[0][1]:
                node = r[1]
                break
        return node

class ProbClassifier:
    def __init__(self):
        self.states_to_nodes = { }
        self.current_state = None
        self.states_to_inputs = { }

    def new_state(self, _state):        #state is input and output, current state is set to state,
                                       #states probabilities are recalculated
        state = str(_state)
        node = None
        if state not in self.states_to_inputs:
            self.states_to_inputs[state] = _state

        if state not in self.states_to_nodes:
            node = ProbNode(state)
            self.states_to_nodes[state] = node
        else:
            node = self.states_to_nodes[state]

        if self.current_state == None:
            self.current_state = node
        else:
            self.current_state.next_hit(node)
            self.current_state = node

        if state not in self.states_to_inputs:
            self.states_to_inputs[state] = _state


    @staticmethod
    def _similarity(str1, str2):
        return SequenceMatcher(None, str1, str2).ratio()

    def apply(self, input_value = None):      #current state set to input value, output is some probability based
                                       #if no input value. based on last current state
                                       #if input value doesn't match any known states, then nearest is found
        state = None
        node = None
        if input_value != None:
            state = str(input_value)
            if state not in self.states_to_nodes:
                similarity = sorted([ [ ProbClassifier._similarity(s, state), s ] for s in self.states_to_nodes ])
                similarity.reverse()
                print "similar to ", state, similarity
                state = similarity[0][1]
                print "Found most similar ", state
            node = self.states_to_nodes[state]
        else:
            node = self.current_state
        new_state = node.next_simulate()
        print "new state:", new_state
        self.current_state = new_state
        return self.states_to_inputs[self.current_state.name]


#gradient ( f(X) ) = [ df/dx0, df/dx1, ..., df/xn ]
#err (f(X), Y) = sum(m:0..M)((Ym - f(Xm))^2)
#as exponential function F(X) has its minimum where dF/dX == 0,
#gradient vector will show the direction to find it, however there is
#a danger for stucking into local optimum, which might occur, therefore
#precision must be set and some way to step over local optimum, like
#increasing step temporarily
#another problem is that system might be stuck into infinite precision
#improving therefore minimum precision E must be set
#we use gradient vector for getting direction towards minimum in N dimensional
#space, which is then multiplied by step variable
# so if F0(W) = sum(x0w0 + x1w0+..+ xnw0)
#gradient (err(F0(W), Y) = sum(m:0,..M)d(Ym^2 - 2*Ym*f(Xm)+f0(Xm)^2)/dW = 
#[ sum(m:0..M)(-2*Ym + 2*(w0*x0)), sum(m:0..M)(-2*Ym + 2*(w0*x1)), .. , sum(m:0..M)(-2*Ym + 2*(w0*xn)) ]
# for F0
#[ sum(m:0..M)(-2*Ym + 2*(w1*x0)), sum(m:0..M)(-2*Ym + 2*(w1*x1)), .. , sum(m:0..M)(-2*Ym + 2*(w1*xn)) ]
# for F1
# and so on
# so for each Fk using gradient vector and step we find wk such that err(Fk(X), Y) is minimal
#i.e next wk = wk + step*gradient(err(Fk(X), Y))


class PolynomClassifier:
    def __init__(self, n_inputs, step = 0.00001, W = 0.1, E = 0.00001, max_iterations = 100000, timeout=4):
        self.n_inputs = n_inputs
        self.X         = [ ]
        self.Y         = [ ]
        self.W         = [ W for i in xrange(0, n_inputs) ]
        self.init_W    = list(self.W)
        self.step      = [ step for i in xrange(0, n_inputs) ]
        self.E         = E
        self.max_iterations = max_iterations
        self.last_grad = None
        self.timeout = timeout

    def _err_func(self, Y, X, W):
        s = 0
        for m in xrange(0, len(X)):
            for k in xrange(0, len(Y[0])):
                s += pow(Y[m][k] - self._func(X[m],W)[k], 2)
        return s

    def _func(self, X, W):
        F = [ ]
        for k in xrange(0, len(self.Y[0])):
            s = 0
            for n in xrange(0, len(X)):
                s += X[n]*W[k]
            F.append(s)
        return F


#(x0wk + ... + xnwk)^2/dwk = (x0wk + ... + xnwk)*(x0wk + ... + xnwk)/dwk = (x0 + ... +xn)*(x0wk+ ...xnwk)*2
    @staticmethod
    def _err_func_k_derivative(Y, Wk, X, k, W):
        s = 0
        for m in xrange(0, len(Y)):
            x1 = 0
            x2 = 0
            for n in xrange(0, len(X[0])):
                x1 += X[m][n]
                x2 += X[m][n]*Wk
            s += -2*Y[m][k]*x1 + 2 * x2 * x1
        return s

    @staticmethod
    def _k_gradient(Y, Wk, X, W, k):
       derivative = PolynomClassifier._err_func_k_derivative(Y, Wk, X, k, W)
       if derivative == 0:
           return 0
       return -1*derivative/abs(derivative)
       #invert gradient as we don't need err function growth direction, but vice versa
       #so now gradient will point towards the minimum
#       return -1*PolynomClassifier._err_func_k_derivative(Y, Wk, X, k, W)

    @staticmethod
    def _gradient(Y, W, X):
        return [ PolynomClassifier._k_gradient(Y, W[k], X, W, k) for k in xrange(0, len(Y[0])) ]

    @staticmethod
    def _multiply_vectors(A, B):
#        print A, B
        return [ A*b for b in B ]

    def _recalc_w(self, step):
        gradient = PolynomClassifier._gradient(self.Y, self.W, self.X)
        self.last_grad = copy.deepcopy(gradient)
#        print "Gradient is ", gradient
        new_W = copy.deepcopy(self.W)
        for k in xrange(0, len(self.Y[0])):
            new_W[k] = self.W[k] + step[k] * gradient[k]
#            print "Set W[",k,"] to ",self.W[k]
        return new_W


    def new_state(self, state):       #state is input and output, each input is weighted to match output
                                      #using using gradient descent and linear regression
        self.X.append(state[:self.n_inputs])
        self.Y.append(state[self.n_inputs:])
        print "Before: X: ", self.X, "Y: ", self.Y, "Func: ", self.apply(self.X[len(self.X)-1])

        print "Err Func: "
        divisor = 5.0
        t = (3.0 - (-2.0))/divisor
        for i in xrange(0, int(divisor)):
            err = self._err_func(self.Y, self.X, [ -2 + t*i for a in xrange(0,len(self.W)) ])
            print "For W ", [ -2 + t*i for a in xrange(0,len(self.W)) ], err
        i = 0
        iteration_counter = 0
        max_iterations = self.max_iterations
        step = copy.deepcopy(self.step)
        self.last_err = last_err = err = self._err_func(self.Y, self.X, self.W)
        print "Initial err: ", err
        print "Initial gradient: ", PolynomClassifier._gradient(self.Y, self.W, self.X)
        time_passed = time.time()
        while err > self.E:
            new_W = self._recalc_w(step)
            if new_W == None:
                break
#            print "Error: ", last_err
#            print "W: ", self.W
#            print "Func: ", self._func(self.X[len(self.X)-1], self.W)
#            print "Gradient: ", self.last_grad

            err = self._err_func(self.Y, self.X, new_W)
            if last_err < err:
                print "New func ", self._func(self.X[len(self.X)-1], new_W)
                print "Last func ", self._func(self.X[len(self.X)-1], self.W)
                print "New err is more than last ", last_err, err
                break

            self.last_err = last_err = err
            self.W = new_W

            gradient = PolynomClassifier._gradient(self.Y, self.W, self.X)

            for u in xrange(0, len(gradient)):
                if self.last_grad[u] * gradient[u] < 0:
                    step[u] = step[u] / 2
                    print "Gradient sign change detected, step decreased", step

            i += 1
            iteration_counter += 1
            if i >= max_iterations/2:
                step = [ step[k] * 2 for k in xrange(0, len(self.W)) ]
                print "Half iterations passed, increasing step to ", step
                i = 0
                max_iterations -= i

            if i >= max_iterations:
                break

            if time.time() - time_passed > self.timeout:
                print "Timeout"
                break


        print "Err now is ", err
        print "W now is ", self.W, " was ", self.init_W
        print "Last gradient was ", self.last_grad
        print "Results: X: ", self.X, "Y: ", self.Y, "Func: ", self.apply(self.X[len(self.X)-1])
        print "Iterations done: ", iteration_counter, "starting step: ", self.step, "final step: ", step

    def apply(self, _input_value):     #output is weighted sum of input
        input_value = None
        if len(_input_value) > self.n_inputs:
            input_value = _input_value[:self.n_inputs]
        else:
            input_value = _input_value
        print "Applying with W ", self.W, " for ", input_value
        res = self._func(input_value, self.W)
        print "len(res) ", len(res), " n_inputs ", self.n_inputs
        while len(res) < self.n_inputs:
            t = copy.deepcopy(res)
            t.extend(res)
            res = t
        return res

class PolynomClassifierChain:
    def __init__(self, num_layers, n_inputs, step = 0.001, W = 0.1, E = 0.00001, max_iterations = 100000, timeout=4):
        self.polynom_classifiers = [ ]
        self.n_inputs = n_inputs
        self.E = E
        for i in xrange(0, num_layers):
            self.polynom_classifiers.append(PolynomClassifier(n_inputs, step,
                                                              W, E, max_iterations, timeout))
    def new_state(self, state):
        print "Polynom layer 0 new state ", state
        self.polynom_classifiers[0].new_state(state)
        if self.polynom_classifiers[0].last_err <= self.E:
            return
        old_res = self.polynom_classifiers[0].apply(state[:self.n_inputs])
        new_state = old_res
        new_state.extend(state[self.n_inputs:])
        for i in xrange(1, len(self.polynom_classifiers)):
            print "Polynom layer ",i," new state ", new_state
            self.polynom_classifiers[i].new_state(new_state)
            old_res = self.polynom_classifiers[i].apply(new_state[:self.n_inputs])
            new_state = old_res
            new_state.extend(state[self.n_inputs:])

    def apply(self, input_value):
        res = input_value
        for i in xrange(0, len(self.polynom_classifiers)):
            if i > 0:
                if self.polynom_classifiers[i].last_err >= self.polynom_classifiers[i-1].last_err:
                    break
            res=self.polynom_classifiers[i].apply(input_value)
            print "Polynom layer ",i," input ", input_value, " output ", res, "last_err: ", self.polynom_classifiers[i].last_err
            if self.polynom_classifiers[i].last_err <= self.E:
                break

            input_value = res
        return res

class DetectClassifier:
    def __init__(self):
        self.polynom_classifiers = [ ]
        self.classifiers_to_states = { }
    def new_state(self, state):      #each state assigned to each class in input
                                     #classes are deduced by using some virtual area mapping
                                     #in state space, if corresponds to multiple classes(overlapping)
                                     #class weighting(prioritization) is used, based
                                     #on how many values of this type were in which class
                                     #in learning samples
        i = self.apply(state[:len(state)-1])
        if i == None:
            p = PolynomClassifier(len(state[:len(state)-1]))
            s = state[:len(state)-1]
            s.append(1)
            p.new_state(s)
            self.polynom_classifiers.append(p)
            i = len(self.polynom_classifiers)-1
            self.classifiers_to_states[i] = state[len(state)-1:]
    def apply(self, input_value):    #input value is state, output - deduced class
        for i in xrange(0,len(self.polynom_classifiers)):
            res = self.polynom_classifiers[i].apply(input_value)
            print res
            if int(round(res[0])) == 1.0:
                val = copy.deepcopy(input_value)
                val.append(self.classifiers_to_states[i])
                return val
        print "Not found"
        return None


class DetectAutoClassifier:
    def __init__(self, class_minsize=1, acc_states = None):
        self.acc_states = acc_states
        self.classes = None
        if self.acc_states == None:
            self.acc_states = [ ]
        self.median = None
        self.classes_dict = { }
        self.class_minsize = class_minsize
        self.class_sizes = { }
        self.min_vec = { }
        self.apply_cnt = 0
    def new_state(self, state):      #each state assigned to each class, class is autogenerated(k_means etc)
        self.acc_states.append(state)
        self._generate_classes()
    def apply(self, input_value):    #input value is state, output - deduced class
        if self.classes == None:
            self._generate_classes()

        self.apply_cnt += 1

        for class_num in self.classes_dict:
            le = DetectAutoClassifier._is_le_than(input_value, self.max_vec[class_num])
            print "le for ", input_value," is ",le
            ge = DetectAutoClassifier._is_ge_than(input_value, self.min_vec[class_num])
            print "ge for ", input_value," is ",ge
            diff = DetectAutoClassifier._get_vec_abs_diff(input_value, self.classes[class_num][0])
            med = DetectAutoClassifier._is_le_than(diff, self.median)
            print "med for ", input_value," is ", med
            if ge == True and le == True:
                self.apply_cnt -= 1
                return class_num

        print input_value," is out of class, generating new classes"
        self.new_state(input_value)
        if self.apply_cnt < 2:
            res = self.apply(input_value)
            self.apply_cnt -= 1
            return res
        self.apply_cnt -= 1
        return None
    def _generate_classes(self):
        self.classes_dict = { }
        self.min_vec = { }
        self.max_vec = { }
        self.acc_states = sorted(self.acc_states)
        self.median = DetectAutoClassifier._get_vec_median(self.acc_states)
        class_num = 0
        self.classes = [ ]
        self.class_sizes = { }
        self.classes.append([ self.acc_states[0], class_num ])
        self.classes_dict[class_num] = [ ]
        self.classes_dict[class_num].append(self.acc_states[0])
        self.class_sizes[class_num]=0
        for i in xrange(1, len(self.acc_states)):
           diff = DetectAutoClassifier._get_vec_abs_diff(self.acc_states[i-1], self.acc_states[i])
           if DetectAutoClassifier._is_le_than(diff, self.median) or self.class_sizes[class_num] < self.class_minsize:
               self.classes.append([ self.acc_states[i], class_num ])
               self.classes_dict[class_num].append(self.acc_states[i])
               self.class_sizes[class_num] += 1
               print "class ", class_num, " for ", self.acc_states[i]
           else:
               class_num += 1
               self.classes_dict[class_num] = [ ]
               self.class_sizes[class_num]=0
               self.classes.append([ self.acc_states[i], class_num ])
               self.classes_dict[class_num].append(self.acc_states[i])
               print "class ", class_num, " for ", self.acc_states[i]

        for class_num in self.classes_dict:
            l = self.classes_dict[class_num]
            if len(l) > 0:
                print l, l[0]
                self.max_vec[class_num] = [ max([ el[j] for el in l ]) for j in xrange(0, len(l[0]))]
                self.min_vec[class_num] = [ min([ el[j] for el in l ]) for j in xrange(0, len(l[0]))]
                print "class ", class_num, " max_vec ", self.max_vec[class_num]
                print "class ", class_num, " min_vec ", self.min_vec[class_num]

        print "Generated classes: ", self.classes
        print "Median ", self.median
    @staticmethod
    def _get_vec_abs_diff(vec1, vec2):
       if isinstance(vec1[0], str):
           return [ ProbClassifier._similarity(v1, v2) for v1,v2 in zip(vec1,vec2) ]
       else:
           return [ abs(v1-v2) for v1,v2 in zip(vec1,vec2) ]
    @staticmethod
    def _get_vec_abs_diff_scalar(vec1, vec2):
       return sum([ abs(v1-v2) for v1,v2 in zip(vec1,vec2) ])
    @staticmethod
    def _is_le_than(vec1, vec2):
       for v1,v2 in zip(vec1,vec2):
           if v1 > v2:
               return False
       return True
    @staticmethod
    def _is_ge_than(vec1, vec2):
       for v1,v2 in zip(vec1,vec2):
           if v1 < v2:
               print "less: ", v1, v2
               return False
       return True
    @staticmethod
    def _get_vec_median(states):
       median = states[0]
       if isinstance(median[0], str):
           #median = [ ProbClassifier._similarity(m, v) for m,v in zip(median,states[0]) ]
           average_similarity = 0
           for i in xrange(1, len(states)):
               median = [ ProbClassifier._similarity(m, v) for m,v in zip(median,states[i]) ]
           print median
           median = [ m/len(states) for m in median ]
       else:
           for i in xrange(1, len(states)):
               median = [ m + v for m,v in zip(median,states[i]) ]
           median = [ m/len(states) for m in median ]
       return median

def testDetectAutoClassifier():
    input_data = [ [0,1], [0,2], [20,30], [30, 40] ]

    c = DetectAutoClassifier(1)

    for i in input_data:
        c.new_state(i)

    v = [ 0, 3 ]
    res = c.apply(v)
    print "classes: ", c.classes
    print "classified ", v, "to class", res
    if res != 0:
        print "classes test not passed0"
        return False

    print "class 0 test passed"

    v = [ 40, 50 ]
    res = c.apply(v)
    print "classes: ", c.classes
    print "classified ", v, "to class", res

    if res != 1:
        print "classes test not passed1"
        return False

    print "class 1 test passed"

    print "All autoclassifier tests passed"

    return True

def testProbClassifier():

    c = ProbClassifier()

    WINTER = 1
    SNOW = 1
    SUMMER = 0
    RAIN = 0
    NO_RAIN_SNOW = 2
    CLOUDY = 1
    SUNNY = 0

    c.new_state([WINTER,SNOW,CLOUDY])
    c.new_state([WINTER,NO_RAIN_SNOW,CLOUDY])
    c.new_state([WINTER,NO_RAIN_SNOW,SUNNY])
    c.new_state([WINTER,RAIN,CLOUDY])
    c.new_state([SUMMER,RAIN,CLOUDY])
    c.new_state([SUMMER,NO_RAIN_SNOW,SUNNY])
    c.new_state([WINTER,RAIN,CLOUDY])
    c.new_state([WINTER,SNOW,CLOUDY])

    res =  c.apply([SUMMER+600,SNOW,CLOUDY])
    if res != [WINTER,NO_RAIN_SNOW,CLOUDY]:
        print "ProbClassifier test not passed0"
        return False

    print "ProbClassfier test passed0"

    res = c.apply()
    if res != [WINTER,NO_RAIN_SNOW,SUNNY]:
        print "ProbClassifier test not passed1"
        return False

    print "ProbClassfier test passed1"

    res = c.apply()
    if res != [WINTER,RAIN,CLOUDY]:
        print "ProbClassifier test not passed2"
        return False

    print "ProbClassfier test passed2"

    print "All ProbClassifier tests passed"

    return True

def testPolynomClassifier():

    c= PolynomClassifier(3)

    c.new_state([ 2,2,2,4,4,4 ])

    c.new_state([ 4,4,4,8,8,8 ])

    res =  c.apply([4,4,4])
    print res
    res = [ math.ceil(r) for r in res ]

    if res != [ 8, 8, 8 ]:
        print "PolynomClassifier test not passed0"
        return False

    print res

    c2 = PolynomClassifier(3)

    c2.new_state([ 4,4,4,12,12,12 ])

    c2.new_state([ 3,3,3,9,9,9 ])

    res =  c2.apply([8,8,8])
    print res
    res = [ math.ceil(r) for r in res ]

    print res
    if res != [ 24, 24, 24 ]:
        print "PolynomClassifier test not passed1"
        return False

    c3 = PolynomClassifier(2)

    c3.new_state([ 200, 200, 1 ])

    res = c3.apply([200, 200])

    print res
    res = [ int(r) for r in res ]

    print res
    if res != [ 1, 1 ]:
        print "PolynomClassifier test not passed2"
        return False


    c4 = PolynomClassifier(1)

    input_data = [ [ 10, 40 ], [ 20, 60 ],  [ 30, 80 ] ]

    for i in input_data:
        c4.new_state(i)

    passed = True

    for i in input_data:
        res = c4.apply(i[:1])
        print res
        res = [ int(r) for r in res ]
        print "result: ", i[:1], res
        print "sample: ", i[:1], i[1:]
        if abs(res[0] - i[1:][0]) > 20:
            print "PolynomClassifier test not passed3"
            passed = False

    if passed == False:
        return False

    print "All Polynom Classifier tests passed"

    return True

def testClassifierApplier():
    c = ClassifierApplier()

    input_data = [ [-2], [0],[1],[2] ]
    output_data = [ [-4], [0],[2],[4] ]

    for i,j in zip(input_data, output_data):
        arr = list(i)
        arr.extend(j)
        c.new_state(arr)

    for i,o in zip(input_data, output_data):
        res = c.apply(i)
        res = [ int(round(r)) for r in res ]

        print res
        if res != o:
            print "Classifier applier test not passed0"
            return False

    print "Classifier applier test passed0"

    c = ClassifierApplier()

    input_data = [ [-2], [0],[1],[2] ]
    output_data = [ [4], [0],[2],[4] ]

    for i,j in zip(input_data, output_data):
        arr = list(i)
        arr.extend(j)
        c.new_state(arr)

    for i,o in zip(input_data, output_data):
        res = c.apply(i)
        res = [ int(round(r)) for r in res ]

        print res
        if res != o:
            print "Classifier applier test not passed1"
            return False

    print "Classifier applier test passed1"

    c = ClassifierApplier()

    print "Polynom input passed"

    input_data = [ [0],[2],[1],[2],[0] ]
    output_data = [ [0],[2],[4],[3],[0] ]

    for i,j in zip(input_data, output_data):
        arr = list(i)
        arr.extend(j)
        c.new_state(arr)

    res = c.apply( input_data[2] )
    res = [ int(r) for r in res ]

    print res
    if res != [ input_data[3][0], output_data[3][0] ]:
        print "Classifier applier test not passed20"
        return False

    print "Classifier applier test passed20"

    res = c.apply( input_data[0])
    res = [ int(r) for r in res ]

    print res
    if res != input_data[0] :
        print "Classifier applier test not passed21"
        return False

    print "Classifier applier test passed21"

    print "Prob input passed"

    c = ClassifierApplier()

    input_data = [ [0.1, 0.1],[20,20],[100,100],[200,200] ]
    output_data = [ [0],[2],[4],[2] ]

    for i,j in zip(input_data, output_data):
        arr = list(i)
        arr.extend(j)
        c.new_state(arr)

    for i,o in zip(input_data, output_data):
        res = c.apply(i)
        print res
#        res = [ int(round(r)) for r in res ]

        print res, o
        if res[2] != o[0]:
            print "Classifier applier test not passed3"
            return False

    print "Classifier applier test passed3"

    print "Detect input passed"

    print "All tests passed for ClassifierApplier"

    return True

def testDetectClassifier():

    c = DetectClassifier()

    input_data = [ [0, 0],[20,20],[100,100],[200,200] ]
    output_data = [ [0],[2],[4],[2] ]

    for i,j in zip(input_data, output_data):
        arr = list(i)
        arr.extend(j)
        c.new_state(arr)


    res = c.apply(input_data[1])

    print res, res[2], output_data[1]
    if res[2] != output_data[1]:
        print "DetectClassifier test not passed0"
        return False

    print "DetectClassifier test passed0"

    print "All tests passed for DetectClassifier"

    return True

def runAllTests():
    if not testClassifierDetector():
        exit(0)
    if not testDetectAutoClassifier():
        exit(0)
    if not testProbClassifier():
        exit(0)
    if not testPolynomClassifier():
        exit(0)
    if not testDetectClassifier():
        exit(0)
    if not testClassifierApplier():
        exit(0)

    print "All tests passed!"


def testChat():

    c = ClassifierApplier()


    c.new_state(["What is life?"])
    c.new_state(["Who am i"])
    c.new_state(["I want to eat"])
    c.new_state(["Should find some food"])
    c.new_state(["Now I fine"])
    c.new_state(["Who am i"])
    c.new_state(["Who are you"])
    c.new_state(["I want to eat"])
    c.new_state(["What is life?"])

    print "results:"

    for i in xrange(0,20):
        print c.apply()

    return True


#def string_number(strings):
#    num_words = 
#    average_letter = 
#    

runAllTests()

#testChat()

exit(0)

#class ClassifierNetworkNode:
#    def __init__(self):
#        self.
#    def new_state(self, state):
#        self.

c = ClassifierApplier()

for i in xrange(0,20):
    c.new_state([ i, math.sin(i) ])
    print [ i, math.sin(i) ]

print "Now approximated:"

for i in xrange(0,20):
    res = c.apply([i])
    print "results:", [ i, res ],"sample values:", [ i, math.sin(i) ]

print c.c.classes
exit(0)


c.new_state([ 0, math.sin(0) ])

for i in xrange(0,10):
    res = c.apply()
    print res

