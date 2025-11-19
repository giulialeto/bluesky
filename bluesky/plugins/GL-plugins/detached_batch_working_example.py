from bluesky import core, stack, traf, tools, settings 
from bluesky.stack.simstack import readscn
import bluesky as bs
import bluesky.plugins.ai4realnet_deploy_RL_tools as RLtools

import debug

def init_plugin():
    deploy_RL = DeployRL()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'Detached_batch_example',
        # The type of this plugin.
        'plugin_type':     'sim',
        }

    return config

class DeployRL(core.Entity):  

    def __init__(self):
        super().__init__()

        self.start_next = True
        self.repeats = 0
        self.total = 0
        self.scentime = []
        self.scencmd = []

    def reset(self):
        pass

    @stack.command
    def detached_batch(self, fname: str):
        """
        Load a batch scenario file, extract REPEATS line,
        and store the scenario commands locally.
        """
        # use the server batch file load function to populate self.scen_batch
        scentime = []
        scencmd = []
        for (cmdtime, cmd) in readscn(fname):
                scentime.append(cmdtime)
                scencmd.append(cmd)
                debug.light_blue(f'[DeployRL] Read command: {cmd} at time {cmdtime}')

        idx = next(
            (i for i, cmd in enumerate(scencmd)
            if cmd.strip().lower().startswith('repeats')),
            None
        )

        if idx is None:
            # No repeats line found
            print(f"[DeployRL] No 'repeats' line found in scenario '{fname}'.")
            return

        cmd = scencmd.pop(idx)
        scentime.pop(idx)
        stack.process(cmd)

        idx = next(
            (i for i, cmd in enumerate(scencmd)
            if cmd.strip().lower().startswith('plugin')),
            None
        )
        scencmd.pop(idx)
        scentime.pop(idx)

        self.scentime = scentime
        self.scencmd = scencmd
        self.start_next = True
        stack.process(f'OP')
        stack.process(f'DTMULT 5000')

    @stack.command
    def end_scen(self):
        """
        Mark the current scenario as finished so the next one can start.
        """
        self.start_next = True

    @stack.command
    def repeats(self, repetitions: int):
        """
        Set how many times the detached batch scenario should be run.
        """
        self.total = self.repeats = repetitions
        
    @core.timed_function(name='update', dt=RLtools.constants.ACTION_FREQUENCY)
    def update(self):
        if self.start_next and self.repeats > 0:
            self.repeats -= 1
            idx_scn = self.total - self.repeats
            bs.sim.start_batch_scenario(f'batch_{idx_scn}', list(self.scentime), list(self.scencmd))
            self.start_next = False
            self.counter = 0



        self.counter +=1

        if self.counter == 20:
            stack.process('END_SCEN')
            if self.repeats == 0:
                stack.stack('QUIT')