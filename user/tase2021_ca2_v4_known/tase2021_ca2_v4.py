#!/usr/bin/env python

# Copyright (C) 2012-2015, Alphan Ulusoy (alphan@bu.edu)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# Case studies presented in:
#
# A. Ulusoy, S. L. Smith, X. C. Ding, C. Belta, D. Rus, "Optimality and
# robustness in multi-robot path planning with temporal logic
# constraints," The International Journal of Robotics Research, vol. 32,
# no. 8, pp. 889-911, 2013.
#
# Note: Case studies 1 and 5 are commented out because they need a lot of memory.

import lomap
from lomap import Ts, Timer
import logging
from collections import namedtuple

# custom packages
import view
import lomap.algorithms.multi_agent_optimal_run_ca as ca

# Logger configuration
logger = logging.getLogger(__name__)

def main():

    Rho = namedtuple('Rho', ['lower', 'upper'])
    rhos = [Rho(lower=0.98, upper=1.04), Rho(lower=0.98, upper=1.04)]

    with Timer('IJRR 2013 Case-Study 2'):
        # norminal
        r1 = Ts.load('./robot_1.yaml')
        r2 = Ts.load('./robot_2.yaml')
        r3 = Ts.load('./robot_3.yaml')                  # robot_3.yaml
        r4 = Ts.load('./robot_4.yaml')                  # robot_4.yaml

        # robustness
        #r1 = Ts.load('./robustness/robot_1.yaml')
        #r2 = Ts.load('./robustness/robot_2.yaml')
        #r3 = Ts.load('./robustness/robot_3.yaml')       # robot_3.yaml
        #r4 = Ts.load('./robustness/robot_4.yaml')

        # CASE 2
        #ts_tuple = (r1, r2)
        #ts_tuple = (r1, r2, r3, r4)
        #formula = ('[]<>gather && [](gather->(r1gather && r2gather)) '
        #           '&& [](r1gather -> X(!r1gather U r1upload)) '
        #           '&& [](r2gather -> X(!r2gather U r2upload))')
        #opt_prop = set(['r1gather','r2gather'])

        # CASE 3
        #ts_tuple = (r1, r2)
        #ts_tuple = (r1, r2, r3, r4)
        #formula = ('[]<>gather && [](gather->(r1gather && r2gather)) '
        #           '&& [](r1gather -> X(!r1gather U r1upload)) '
        #           '&& [](r2gather -> X(!r2gather U r2upload)) '
        #           '&& [](!(r1gather1 && r2gather1) && !(r1gather2 && r2gather2)'
        #           '&& !(r1gather3 && r2gather3) && !(r1gather4 && r2gather4))')
        #opt_prop = set(['r1gather','r2gather'])

        # CASE 4
        #ts_tuple = (r1, r2)
        ts_tuple = (r1, r2, r3, r4)
        formula = ('[]<>gather && [](gather->(r1gather2 && r2gather4)) '
                   '&& [](r1gather -> X(!r1gather U r1upload1)) '
                   '&& [](r2gather -> X(!r2gather U r2upload2))')
        opt_prop = set(['r1gather2','r2gather4'])

        # collision avoidance
        is_modifible = [True, True, False, False]
        logger.info('Formula: %s', formula)
        logger.info('opt_prop: %s', opt_prop)


        #prefix_length_pre, prefixes, suffix_cycle_cost_pre, suffix_cycles, team_prefix, team_suffix_cycle = \
        #    ca.multi_agent_optimal_run_ca_pre(ts_tuple, formula, opt_prop, is_pp=True)

        prefix_length, prefixes, suffix_cycle_cost, suffix_cycles, team_prefix, team_suffix_cycle = \
            ca.multi_agent_optimal_run_ca(ts_tuple, formula, opt_prop, is_modifible, is_pp=False)
        #prefix_length, prefixes, suffix_cycle_cost, suffix_cycles, team_prefix, team_suffix_cycle = \
        #    ca.multi_agent_optimal_run(ts_tuple, formula, opt_prop)

        #logger.info('[Pre-deleted] Cost: %d', suffix_cycle_cost_pre)
        #logger.info('[Pre-deleted] Prefix length: %d', prefix_length_pre)
        logger.info('Cost: %d', suffix_cycle_cost)
        logger.info('Prefix length: %d', prefix_length)
        # Find the controls that will produce this run
        control_prefixes = []
        control_suffix_cycles = []
        for i in range(0, len(ts_tuple)):
            ts = ts_tuple[i]
            control_prefixes.append(ts.controls_from_run(prefixes[i]))
            control_suffix_cycles.append(ts.controls_from_run(suffix_cycles[i]))
            logger.info('%s run prefix: %s', ts.name, prefixes[i])
            logger.info('%s control perfix: %s', ts.name, control_prefixes[i])
            logger.info('%s suffix cycle: %s', ts.name, suffix_cycles[i])
            logger.info('%s control suffix cycle: %s', ts.name,
                                                    control_suffix_cycles[i])
    # visualize run
    #view.visualize_run(r1, suffix_cycles[0])
    #view.visualize_run(r1, suffix_cycles[1])
    #view.visualize_run(r1, suffix_cycles[2])
    #view.visualize_run(r2, suffix_cycles[1])

    # animations
    view.visualize_animation_w_team_run(ts_tuple, team_suffix_cycle)

    logger.info('<><><> <><><> <><><>')


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
    #formatter = logging.Formatter(
#                         '%(levelname)s %(name)s %(asctime)s - %(message)s',
#                         datefmt='%m/%d/%Y %H:%M:%S')
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
