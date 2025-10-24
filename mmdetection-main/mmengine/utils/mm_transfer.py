
class MMTransfer(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

def record_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            mm_transfer.temp['grads'][name] = param.grad.detach().clone()

mm_transfer = MMTransfer()
mm_transfer.temp = dict(grads=dict())
mm_transfer.resumed = dict()
mm_transfer.record_grads = record_grads
