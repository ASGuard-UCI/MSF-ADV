import platform
import yaml
import re
import warnings
import os
import uuid
import multiprocessing
import getpass
import sys
import logging

import subprocess
from pdb import set_trace as st
# Map hostname patterns to human-readable names
__config_file_mapping = {
    r'(flux-login\d+|nyx\d+)\.arc-ts\.umich\.edu' : 'flux',
    r'(compute-[0-9]+\.)?vl-fb\.eecs.umich\.edu' : 'vl-fb',
    r'\w+\.dc\.umich\.edu': 'caen',
    r'[\w-]+\.engin\.umich\.edu': 'caen',
    r'pinwheel[0-9]+\.d1\.comp\.nus\.edu\.sg': 'pinwheel',
    r'pinwheel[0-9]' : 'pinwheel',
    r'head\.ionic\.cs\.princeton\.edu': 'ionic',
    r'node\d+\.ionic\.cs\.princeton\.edu': 'ionic',
    r'haystack-gpu': 'haystack-gpu'
}

MBII_DATASET_LAB_DIR = None # The 'lab' directory of the MIT-Berkeley Dataset.
SIRFS_DIR = None
PASCAL3D_ROOT = None
SHAPENET_DIR = None

# The directory to save intermediate files which can be safely deleted
TMP_DIR = None

# The command to run mitsuba.
# Is a list, for example, ['mitusab', '-q']
MITSUBA_COMMAND = None

# The paths to append in order to run mitsuba
# Is a dictionary, for example: {'PATH': '/bin', 'LD_LIBRARY_PATH': '/lib'}
MITSUBA_APPEND_PATH = None

TENSORBOARD_LOGDIR = None

RENDER4CNN_WEIGHTS = None
FT_RENDER4CNN_WEIGHTS = None
CLICKHERE_CNN_WEIGHTS = None

SIBL_COMPUTED_SPH_COEFF = None
SYNTHETIC_OFFLINE_DIR = None
SAVE_PATH=None
LOAD_PATH=None

# Load corresponding configure file using hostname
hostname = platform.node()
username = getpass.getuser()
try:
    basename = next(bname for regex, bname in __config_file_mapping.items() if re.match(regex, hostname))
except StopIteration:
    print('Machine not configured: {}'.format(hostname))
    exit(1)

BASENAME = basename
# print('Detected platform: {}'.format(basename))
filename = '{}.yml'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)), basename, username))

with open(filename, 'r') as f:
    variables = yaml.load(f)

if not variables:
    warnings.warn('No variables in {}.'.format(filename))

unused_vars = set(variables.keys()).difference(set(globals().keys()))
if len(unused_vars) > 0:
    warnings.warn('The folllowing defined variables are not used:')
    for v in unused_vars: print(v)

globals().update(variables)

# Obtain unique ID:
if basename in ['ionic', 'vl-fb']: # Slurm
    UNIQUE_ID = os.environ.get('SLURM_JOBID') or uuid.uuid4().hex
elif basename =='flux': # PBS
    UNIQUE_ID = os.environ.get('PBS_JOBID') or uuid.uuid4().hex
    if not UNIQUE_ID: UNIQUE_ID = uuid.uuid4().hex
    else: UNIQUE_ID = UNIQUE_ID.split('.')[0]
elif basename in ['v6', 'laptop', 'caen', 'pinwheel', 'haystack-gpu']:
    UNIQUE_ID = uuid.uuid4().hex
else:
    raise NotImplementedError


# Get number of cores
NUM_CORES = multiprocessing.cpu_count()


def check_job_flux(job_id):
    try:
        ps = subprocess.Popen(['qstat', '-f', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = subprocess.check_output(['grep',  'job_state = R'], stdin=ps.stdout)
        logging.info('Checking %s: success: %s', job_id, out.decode())
        return 0
    except subprocess.CalledProcessError as e:
        logging.info('Checking %s: failed: (%d) %s', job_id, e.returncode, ps.stderr.read())

        return e.returncode


def check_job_vlfb(job_id):
    try:
        ps = subprocess.Popen(['scontrol', 'show', 'jobid', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = subprocess.check_output(['grep', 'JobState=RUNNING'], stdin=ps.stdout)
        logging.info('Checking %s: success: %s', job_id, out.decode())
        return 0
    except subprocess.CalledProcessError as e:
        logging.info('Checking %s: failed: (%d) %s', job_id, e.returncode, ps.stderr.read())

        return e.returncode


__CHECK_JOB_COMMAND = {
    'flux' : check_job_flux,
    'vl-fb' : check_job_vlfb
}


def check_job(job):
    if job == 'debug':
        return True
    # assert str.isdigit(job)
    return __CHECK_JOB_COMMAND[BASENAME](job) == 0
