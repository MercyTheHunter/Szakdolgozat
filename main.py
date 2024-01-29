import conv_nn as c
import functions as func

train_loader, valid_loader, test_loader = func.make_loaders(batch_size=256)
model = c.Conv_NN()

#func.example_plot(train_loader=train_loader)

model, train_losses, valid_losses = func.train_model(model=model,
                                                     train_loader=train_loader,
                                                     valid_loader=valid_loader,
                                                     patience=5,
                                                     n_epochs=100)

func.count_parameters(model)

func.loss_plot(train_loss=train_losses, valid_loss=valid_losses)

func.evaluate_model(model=model, test_loader=test_loader, batch_size=256)

func.sample_test(model=model, test_loader=test_loader)