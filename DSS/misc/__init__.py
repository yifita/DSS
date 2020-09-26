import time
import threading
from .. import logger_py


class Thread(threading.Thread):
    def __init__(self, target, name='', args=(), kwargs={}):
        super().__init__(target=target, name=name, args=args, kwargs=kwargs)
        self.args = args
        self.kwargs = kwargs
        self.name

    def run(self):
        t0 = time.time()
        super().run()
        t1 = time.time()
        logger_py.info('{}: {:.3f} seconds'.format(self.name, t1 - t0))
