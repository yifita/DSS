from __future__ import division

import sys
import time


class ansi:
    """Color codes to print with color"""
    WHITE = '\033[0;97m'
    WHITE_B = '\033[1;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'


def error(message, *lines, ostream=sys.stdout):
    string = "{}{}: " + message + ("{}\n" if lines else
                                   "{}") + "\n".join(lines) + "{}"
    print(
        string.format(ansi.RED_B,
                      time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                      ansi.RED, ansi.ENDC), file=ostream)
    sys.exit(-1)


def warn(message, *lines, ostream=sys.stdout):
    if message:
        string = "{}{}: " + message + ("{}\n" if lines else
                                       "{}") + "\n".join(lines) + "{}"
        print(
            string.format(ansi.YELLOW_B,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                          ansi.YELLOW, ansi.ENDC), file=ostream)
    else:
        string = "{}{}: " + "\n".join(lines) + "{}"
        print(
            string.format(ansi.YELLOW,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                          ansi.ENDC), file=ostream)


def info(message, *lines, ostream=sys.stdout, bold=False):
    if bold:
        color = ansi.WHITE_B
    else:
        color = ansi.WHITE
    if message:
        string = "{}{}: " + message + ("{}\n" if lines else
                                       "{}") + "\n".join(lines) + "{}"
        print(
            string.format(color,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                          ansi.WHITE, ansi.ENDC), file=ostream)
    else:
        string = "{}{}: " + "\n".join(lines) + "{}"
        print(
            string.format(ansi.WHITE,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                          ansi.ENDC), file=ostream)


def success(message, *lines, ostream=sys.stdout):
    if message:
        string = "{}{}: " + message + ("{}\n" if lines else
                                       "{}") + "\n".join(lines) + "{}"
        print(
            string.format(ansi.BLUE_B,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                          ansi.BLUE, ansi.ENDC), file=ostream)
    else:
        string = "{}{}: " + "\n".join(lines) + "{}"
        print(
            string.format(ansi.BLUE,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                          ansi.ENDC), file=ostream)
