import conv_nn as c
import functions as func
import os

train_loader, valid_loader, test_loader = func.make_loaders(batch_size=256)

func.example_plot(train_loader=train_loader)

if os.path.isfile('./conv_nn.pkl'):
    model = func.load_model('conv_nn.pkl')
else:
    model = c.Conv_NN()
    model, train_losses, valid_losses = func.train_model(model=model,
                                                     train_loader=train_loader,
                                                     valid_loader=valid_loader,
                                                     patience=5,
                                                     n_epochs=100)
    func.save_model(model, "conv_nn.pkl")
    func.loss_plot(train_loss=train_losses, valid_loss=valid_losses)

func.count_parameters(model)

func.evaluate_model(model=model, test_loader=test_loader, batch_size=256)

func.sample_test(model=model, test_loader=test_loader)

#func.print_confusion_matrix(model=model, train_loader=train_loader, test_loader=test_loader)