class Trainer:
  def __init__(self, model=None, training_args=None, data_loader=None):
    self.args = training_args
    self.model = model
    self.data_loader = data_loader
    self.loss = nn.NLLLoss()
  
  def train_step(self, input, label):
    hidden = self.model.init_hidden()
    self.model.zero_grad()
    
    for i in range(input.size()[0]):
      output, hidden = self.model(input[i], hidden)
    
    loss = self.loss(output, label)
    loss.backward()

    for p in self.model.parameters():
      p.data.add_(p.grad.data, alpha=-self.args.learning_rate)
    
    return output, loss.item()

  def train(self):
    for batch in range(self.args.num_batch):
      current_loss = 0
      for input, label in self.data_loader:
        output, loss = self.train_step(input, label)
        current_loss += loss
      print(current_loss)

def main():
  model = 
  trainer = Trainer()

if __name__ == "__main__":
  main()