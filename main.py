import conv_nn as c
import functions as func

train_loader, test_loader = func.make_loaders()
model = c.Conv_NN()

print(f"Our model has {func.count_parameters(model)} trainable parameters\n")

func.train_model(model=model, train_loader=train_loader)

func.evaluate_model(model=model, test_loader=test_loader)