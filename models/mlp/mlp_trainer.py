def train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, num_epochs, cuda=False):

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, loss_fn, train_dl, cuda=cuda)
        val_loss = evaluate(model, loss_fn, val_dl, cuda=cuda)
        is_best = val_loss <= best_val_loss
        if is_best:
            best_val_loss = val_loss

def train(model, optimizer, loss_fn, train_dl, cuda=False):
    model.train()

    train_loss = RunningAverage()
    for train_batch, target_batch in train_dl:
        if cuda:
            train_batch, target_batch = train_batch.cuda(non_blocking=True), target_batch.cuda(non_blocking=True)
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item())
    return train_loss()

def evaluate(model, loss_fn, val_dl, cuda=False):
    model.eval()

    eval_loss = RunningAverage()
    for eval_batch, target_batch in val_dl:
        if cuda:
            eval_batch, target_batch = eval_batch.cuda(non_blocking=True), target_batch.cuda(non_blocking=True)
        output_batch = model(eval_batch)
        loss = loss_fn(output_batch, target_batch)
        eval_loss.update(loss.item())
    return eval_loss()

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)