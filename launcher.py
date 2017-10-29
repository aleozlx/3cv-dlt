from __future__ import print_function
from __future__ import unicode_literals
import sys, os

DOCKER="nvidia-docker"
IMAGE="aleozlx/cvstack"
PORTS = []
VOLUMNS = [
    '-v', os.path.abspath('.') + ':/home/developer/workspace:ro',
    '-v', '/tmp/.X11-unix:/tmp/.X11-unix:rw',
    '-v', os.path.expanduser('~/.Xauthority') + ':/home/developer/.Xauthority:ro'
]
argv = [DOCKER, 'run', '-it', '-e', 'DISPLAY', '--net=host'] + PORTS + VOLUMNS + [IMAGE] + ['python3', 'homography.py']
print(' '.join(map(lambda i: ("'%s'"%i) if ' ' in i else i, argv)))
os.execvp(DOCKER, argv)
