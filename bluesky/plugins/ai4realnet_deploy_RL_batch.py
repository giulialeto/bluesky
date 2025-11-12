from bluesky.stack.simstack import readscn
from bluesky import stack
import bluesky as bs


def __init__(self):
    self.start_next = True
    self.repeats = 0
    self.total = 0
    self.scentime = []
    self.scencmd = []

def reset(self):
    pass

@stack.command
def detached_batch(self, fname):
    # use the server batch file load function to populate self.scen_batch
    scentime = []
    scencmd = []
    for (cmdtime, cmd) in readscn(fname):
            scentime.append(cmdtime)
            scencmd.append(cmd)

    idx = next(i for i, cmd in enumerate(scencmd) if cmd.lower().startswith('repeats'))
    cmd = scencmd.pop(idx)
    scentime.pop(idx)
    stack.process(cmd)
    self.scentime = scentime
    self.scencmd = scencmd
    self.start_next = True

@stack.command
def end_scen(self):
     self.start_next = True

@stack.command
def repeats(self, repetitions: int):
     self.total = self.repeats = repetitions
     

def update(self):
    if self.start_next:
        self.repeats -= 1
        bs.sim.start_batch_scenario(f'batch_{self.total - self.repeats}', list(self.scentime), list(self.scencmd))
        