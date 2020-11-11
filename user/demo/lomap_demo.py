#!/usr/bin/env python
import lomap
import lomap.algorithms.multi_agent_optimal_run_ca as ca
import sys
import logging

# Classes derived from namedtuple
from collections import namedtuple
Rho = namedtuple('Rho', ['lower', 'upper'])

def main():
  with lomap.Timer('File I/O'):
    r1 = lomap.Ts()
    r2 = lomap.Ts()

    # Case-Study
    r1 = lomap.Ts.load('robot_1.yaml')
    r2 = lomap.Ts.load('robot_2.yaml')
    r3 = lomap.Ts.load('robot_3.yaml')
    formula = '[](gather1 -> X(!gather1 U upload1)) && [](gather2 -> X(!gather2 U upload2))  && [](gather3 -> X(!gather3 U upload3)) && []<> gather'
    opt_prop = set(['gather'])

  with lomap.Timer('DARS 2012 Case-Study'):
    ts_tuple = (r1, r2, r3)
    # deviation values of agents
    rhos = [Rho(lower=0.95, upper=1.05), Rho(lower=0.95, upper=1.05)]
    #prefix_length, prefixes, suffix_cycle_cost, suffix_cycles = lomap.robust_multi_agent_optimal_run(ts_tuple, rhos, formula, opt_prop)
    prefix_length, prefixes, suffix_cycle_cost, suffix_cycles = ca.multi_agent_optimal_run(ts_tuple, formula, opt_prop)
    print "Cost: %d" % suffix_cycle_cost
    print "Prefix length: %d" % prefix_length

def config_debug():
  # create root logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  # create file handler
  fh = logging.FileHandler('lomap.log', mode='w')
  fh.setLevel(logging.DEBUG)
  # create console handler
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  # create formatter and add it to the handlers
  formatter = logging.Formatter('%(levelname)s %(name)s - %(message)s')
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  # add the handlers to the logger
  logger.addHandler(fh)
  logger.addHandler(ch)
  logger.debug('Logger initialization complete.')

if __name__ == '__main__':
  config_debug()
  main()