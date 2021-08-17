import more_itertools
import contextlib
import pandas
import sys
import numpy as np
from itertools import product
import datetime
import os

class DateIter () : 
    def __init__ (self, start, end) :
        self.start = start
        self.end = end
    
    def __iter__ (self) : 
        return self

    def __next__ (self) : 
        curr = self.start
        if self.start.date != self.end.date :
            self.start = self.start + 1
            return curr
        else : 
            raise StopIteration()

class Date () : 

    MONTHS = ['Jan', 'Feb', 'Mar', 
            'Apr', 'May', 'Jun', 
            'Jul', 'Aug', 'Sep', 
            'Oct', 'Nov', 'Dec']
    
    def __init__ (self, date) : 
        self.date = date
        d, m  = date.split(' ')
        d = int(d)
        self.day = d
        self.month = self.MONTHS.index(m) + 1

    def __add__ (self, n) : 
        td = datetime.timedelta(days=n)
        newDate = datetime.date(2020, self.month, self.day) + td
        month = self.MONTHS[newDate.month - 1]
        day = newDate.day
        return Date(f'{day} {month}')
    
    def __sub__ (self, that) :
        d1 = datetime.date(2020, self.month, self.day)
        d2 = datetime.date(2020, that.month, that.day)
        return (d1 - d2).days

    def __lt__ (self, that) : 
        return (self.month, self.day) < (that.month, that.day)
        
    def __gt__ (self, that) : 
        return (self.month, self.day) > (that.month, that.day)
    
    def __le__ (self, that) : 
        return (self.month, self.day) <= (that.month, that.day)
        
    def __ge__ (self, that) : 
        return (self.month, self.day) >= (that.month, that.day)

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def getInfectedAndRecovered(csvFile):
    data = pandas.read_csv(csvFile)
    confirmed = data['Total Cases']
    recovered = data['Total Recoveries']
    dead      = data['Total Deaths']
    r = recovered + dead
    i = confirmed - r
    return pandas.concat((i, r), axis=1).to_numpy()

def sortAndFlattenDict(d) : 
    return list(unzip(sorted(d.items()))[1])

def dictProduct (d) : 
    return map(dict, product(*map(lambda x : product([x[0]], x[1]), d.items())))

