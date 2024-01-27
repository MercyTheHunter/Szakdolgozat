import conv_nn as c
import functions as func

train_loader, valid_loader, test_loader = func.make_loaders(batch_size=256)
model = c.Conv_NN()

print(f"Our model has {func.count_parameters(model)} trainable parameters\n")

model, avg_train_losses, avg_valid_losses = func.train_model(model=model,
                                                             train_loader=train_loader,
                                                             valid_loader=valid_loader,
                                                             patience=10,
                                                             n_epochs=100)

func.evaluate_model(model=model, test_loader=test_loader)