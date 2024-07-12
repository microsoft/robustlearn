import random
from auto_LiRPA.utils import logger

class BaseScheduler(object):
    def __init__(self, max_eps, opt_str):
        self.parse_opts(opt_str)
        self.prev_loss = self.loss = self.max_eps = self.epoch_length = float("nan")
        self.eps = 0.0
        self.max_eps = max_eps
        self.is_training = True
        self.epoch = 0
        self.batch = 0

    def __repr__(self):
        return '<BaseScheduler: eps {}, max_eps {}>'.format(self.eps, self.max_eps)

    def parse_opts(self, s):
        opts = s.split(',')
        self.params = {}
        for o in opts:
            if o.strip():
                key, val = o.split('=')
                self.params[key] = val

    def get_max_eps(self):
        return self.max_eps

    def get_eps(self):
        return self.eps

    def reached_max_eps(self):
        return abs(self.eps - self.max_eps) < 1e-3

    def step_batch(self, verbose=False):
        if self.is_training:
            self.batch += 1
        return

    def step_epoch(self, verbose=False):
        if self.is_training:
            self.epoch += 1
        return

    def update_loss(self, new_loss):
        self.prev_loss = self.loss
        self.loss = new_loss

    def train(self):
        self.is_training = True
        
    def eval(self):
        self.is_training = False
    
    # Set how many batches in an epoch
    def set_epoch_length(self, epoch_length):
        self.epoch_length = epoch_length


class FixedScheduler(BaseScheduler):
    def __init__(self, max_eps, opt_str=""):
        super(FixedScheduler, self).__init__(max_eps, opt_str)
        self.eps = self.max_eps


class LinearScheduler(BaseScheduler):

    def __init__(self, max_eps, opt_str):
        super(LinearScheduler, self).__init__(max_eps, opt_str)
        self.schedule_start = int(self.params['start'])
        self.schedule_length = int(self.params['length'])
        self.epoch_start_eps = self.epoch_end_eps = 0

    def __repr__(self):
        return '<LinearScheduler: start_eps {:.3f}, end_eps {:.3f}>'.format(
            self.epoch_start_eps, self.epoch_end_eps)

    def step_epoch(self, verbose = True):
        self.epoch += 1
        self.batch = 0
        if self.epoch < self.schedule_start:
            self.epoch_start_eps = 0
            self.epoch_end_eps = 0
        else:
            eps_epoch = self.epoch - self.schedule_start
            if self.schedule_length == 0:
                self.epoch_start_eps = self.epoch_end_eps = self.max_eps
            else:
                eps_epoch_step = self.max_eps / self.schedule_length
                self.epoch_start_eps = min(eps_epoch * eps_epoch_step, self.max_eps)
                self.epoch_end_eps = min((eps_epoch + 1) * eps_epoch_step, self.max_eps)
        self.eps = self.epoch_start_eps
        if verbose:
            logger.info("Epoch {:3d} eps start {:7.5f} end {:7.5f}".format(self.epoch, self.epoch_start_eps, self.epoch_end_eps))

    def step_batch(self):
        if self.is_training:
            self.batch += 1
            eps_batch_step = (self.epoch_end_eps - self.epoch_start_eps) / self.epoch_length
            self.eps = self.epoch_start_eps + eps_batch_step * (self.batch - 1)
            if self.batch > self.epoch_length:
                logger.warning('Warning: we expect {} batches in this epoch but this is batch {}'.format(self.epoch_length, self.batch))
                self.eps = self.epoch_end_eps

class RangeScheduler(BaseScheduler):

    def __init__(self, max_eps, opt_str):
        super(RangeScheduler, self).__init__(max_eps, opt_str)
        self.schedule_start = int(self.params['start'])
        self.schedule_length = int(self.params['length'])

    def __repr__(self):
        return '<RangeScheduler: epoch [{}, {}]>'.format(
            self.schedule_start, self.schedule_start + self.schedule_length)

    def step_epoch(self, verbose = True):
        self.epoch += 1
        if self.epoch >= self.schedule_start and self.epoch < self.schedule_start + self.schedule_length:
            self.eps = self.max_eps
        else:
            self.eps = 0

    def step_batch(self):
        pass

class BiLinearScheduler(LinearScheduler):

    def __init__(self, max_eps, opt_str):
        super(BiLinearScheduler, self).__init__(max_eps, opt_str)
        self.schedule_start = int(self.params['start'])
        self.schedule_length = int(self.params['length'])
        self.schedule_length_half = self.schedule_length / 2
        self.epoch_start_eps = self.epoch_end_eps = 0

    def __repr__(self):
        return '<BiLinearScheduler: start_eps {:.5f}, end_eps {:.5f}>'.format(
            self.epoch_start_eps, self.epoch_end_eps)        
    
    def step_epoch(self, verbose = True):
        self.epoch += 1
        self.batch = 0
        if self.epoch < self.schedule_start:
            self.epoch_start_eps = 0
            self.epoch_end_eps = 0
        else:
            eps_epoch = self.epoch - self.schedule_start
            eps_epoch_step = self.max_eps / self.schedule_length_half
            if eps_epoch < self.schedule_length_half:
                self.epoch_start_eps = min(eps_epoch * eps_epoch_step, self.max_eps)
                self.epoch_end_eps = min((eps_epoch + 1) * eps_epoch_step, self.max_eps)
            else:
                self.epoch_start_eps = max(0, 
                    self.max_eps - ((eps_epoch - self.schedule_length_half) * eps_epoch_step))
                self.epoch_end_eps = max(0, self.epoch_start_eps - eps_epoch_step)
        self.eps = self.epoch_start_eps
        if verbose:
            logger.info("Epoch {:3d} eps start {:7.5f} end {:7.5f}".format(self.epoch, self.epoch_start_eps, self.epoch_end_eps))


class SmoothedScheduler(BaseScheduler):

    def __init__(self, max_eps, opt_str):
        super(SmoothedScheduler, self).__init__(max_eps, opt_str)
        # Epoch number to start schedule
        self.schedule_start = int(self.params['start'])
        # Epoch length for completing the schedule
        self.schedule_length = int(self.params['length'])
        # Mid point to change exponential to linear schedule
        self.mid_point = float(self.params.get('mid', 0.25))
        # Exponential
        self.beta = float(self.params.get('beta', 4.0))
        assert self.beta >= 2.
        assert self.mid_point >= 0. and self.mid_point <= 1.
        self.batch = 0

    
    # Set how many batches in an epoch
    def set_epoch_length(self, epoch_length):
        if self.epoch_length != self.epoch_length:
            self.epoch_length = epoch_length
        else:
            if self.epoch_length != epoch_length:
                raise ValueError("epoch_length must stay the same for SmoothedScheduler")

    def step_epoch(self, verbose = True):
        super(SmoothedScheduler, self).step_epoch()
        # FIXME 
        if verbose == False:
            for i in range(self.epoch_length):
                self.step_batch()
            
    # Smooth schedule that slowly morphs into a linear schedule.
    # Code is based on DeepMind's IBP implementation:
    # https://github.com/deepmind/interval-bound-propagation/blob/2c1a56cb0497d6f34514044877a8507c22c1bd85/interval_bound_propagation/src/utils.py#L84
    def step_batch(self, verbose=False):
        if self.is_training:
            self.batch += 1
            init_value = 0.0
            final_value = self.max_eps
            beta = self.beta
            step = self.batch - 1
            # Batch number for schedule start
            init_step = (self.schedule_start - 1) * self.epoch_length
            # Batch number for schedule end
            final_step = (self.schedule_start + self.schedule_length - 1) * self.epoch_length
            # Batch number for switching from exponential to linear schedule
            mid_step = int((final_step - init_step) * self.mid_point) + init_step
            t = (mid_step - init_step) ** (beta - 1.)
            # find coefficient for exponential growth, such that at mid point the gradient is the same as a linear ramp to final value
            alpha = (final_value - init_value) / ((final_step - mid_step) * beta * t + (mid_step - init_step) * t)
            # value at switching point
            mid_value = init_value + alpha * (mid_step - init_step) ** beta
            # return init_value when we have not started
            is_ramp = float(step > init_step)
            # linear schedule after mid step
            is_linear = float(step >= mid_step)
            exp_value = init_value + alpha * float(step - init_step) ** beta
            linear_value = min(mid_value + (final_value - mid_value) * (step - mid_step) / (final_step - mid_step), final_value)
            self.eps = is_ramp * ((1.0 - is_linear) * exp_value + is_linear * linear_value) + (1.0 - is_ramp) * init_value

class AdaptiveScheduler(BaseScheduler):
    def __init__(self, max_eps, opt_str):
        super(AdaptiveScheduler, self).__init__(max_eps, opt_str)
        self.schedule_start = int(self.params['start'])
        self.min_eps_step = float(self.params.get('min_step', 1e-9))
        self.max_eps_step = float(self.params.get('max_step', 1e-4))
        self.eps_increase_thresh = float(self.params.get('increase_thresh', 1.0))
        self.eps_increase_factor = float(self.params.get('increase_factor', 1.5))
        self.eps_decrease_thresh = float(self.params.get('decrease_thresh', 1.5))
        self.eps_decrease_factor = float(self.params.get('decrease_factor', 2.0))
        self.small_loss_thresh = float(self.params.get('small_loss_thresh', 0.05))
        self.epoch = 0
        self.eps_step = self.min_eps_step
    
    def step_batch(self):
        if self.eps < self.max_eps and self.epoch >= self.schedule_start and self.is_training:
            if self.loss != self.loss or self.prev_loss != self.prev_loss:
                # First 2 steps. Use min eps step
                self.eps += self.min_eps_step
            else:
                # loss decreasing or loss very small. Increase eps step
                if self.loss < self.eps_increase_thresh * self.prev_loss or self.loss < self.small_loss_thresh:
                    self.eps_step = min(self.eps_step * self.eps_increase_factor, self.max_eps_step)
                # loss increasing. Decrease eps step
                elif self.loss > self.eps_decrease_thresh * self.prev_loss:
                    self.eps_step = max(self.eps_step / self.eps_decrease_factor, self.min_eps_step)
                # print("loss {:7.5f} prev_loss {:7.5f} eps_step {:7.5g}".format(self.loss, self.prev_loss, self.eps_step))
                # increase eps according to loss
                self.eps = min(self.eps + self.eps_step, self.max_eps)
            # print("eps step size {:7.5f}, eps {:7.5f}".format(self.eps_step, self.eps))


if __name__ == "__main__":
    s = SmoothedScheduler(0.1, "start=2,length=10,mid=0.3")
    epochs = 20
    batches = 10
    loss = 1.0
    eps = []
    s.set_epoch_length(batches)
    for epoch in range(1,epochs+1):
        s.step_epoch()
        for batch in range(1,batches+1):
            s.step_batch()
            loss = loss * (0.975 + random.random() / 20)
            eps.append(s.get_eps())
            print('epoch {:5d} batch {:5d} eps {:7.5f} loss {:7.5f}'.format(epoch, batch, s.get_eps(), loss))
            # update_loss is only necessary for adaptive eps scheduler
            s.update_loss(loss)
    # plot epsilon values
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,8))
    plt.plot(eps)
    plt.xticks(range(0, epochs*batches+batches, batches))
    plt.grid()
    plt.tight_layout()
    plt.savefig('epsilon.pdf')

