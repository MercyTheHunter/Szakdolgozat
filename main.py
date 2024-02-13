import conv_nn as c
import functions as func
import os

dataset = "MNIST"
filename = dataset + "_conv_nn.pkl"
current_path = os.path.dirname(os.path.realpath(__file__))

if dataset == "MNIST":
    train_loader, valid_loader, test_loader = func.MNIST_make_loaders(batch_size=256)
    func.example_plot(train_loader=train_loader)
elif dataset == "FashionMNIST":
    train_loader, valid_loader, test_loader = func.FashionMNIST_make_loaders(batch_size=256)
    func.example_plot(train_loader=train_loader)
elif dataset == "STL10":
    train_loader, valid_loader, test_loader = func.STL10_make_loaders(batch_size=256)
    #NEEDS FIXING
    func.example_plot(train_loader=train_loader)

if os.path.isfile(os.path.join(current_path, "SavedModels/", filename)):
    model = func.load_model(filename)
else:
    model = c.Conv_NN()
    model, train_losses, valid_losses = func.train_model(model=model,
                                                     train_loader=train_loader,
                                                     valid_loader=valid_loader,
                                                     patience=5,
                                                     n_epochs=100)
    func.save_model(model, filename)
    func.loss_plot(train_loss=train_losses, valid_loss=valid_losses)

func.count_parameters(model)

func.evaluate_model(model=model, test_loader=test_loader, batch_size=256)

func.sample_test(model=model, test_loader=test_loader)
