import models as m
import functions as func
import os

patience = 3 #3 or 5 or 7 (for save location)
kernel = 7 #3 or 5 or 7 (for save location)
model = 4 #1:CNN_small, 2:CNN_medium, 3:FNN_small, 4:FNN_medium
classes = 10 #MNIST: 10, FashionMNIST: 10, CATDOG: 2

#dataset = "MNIST"
dataset = "FashionMNIST"
#dataset = "CATDOG"

if model == 1:
    filename = dataset + "_CNN_small"
elif model == 2:
    filename = dataset + "_CNN_medium"
elif model == 3:
    filename = dataset + "_FNN_small"
elif model == 4:
    filename = dataset + "_FNN_medium"

if model == 1:
    savedmodelname = filename + ".pkl"
elif model == 2:
    savedmodelname = filename + ".pkl"
elif model == 3:
    savedmodelname = filename + ".pkl"
elif model == 4:
    savedmodelname = filename + ".pkl"

current_path = os.path.dirname(os.path.realpath(__file__))

if dataset == "MNIST":
    train_loader, valid_loader, test_loader = func.MNIST_make_loaders(batch_size=256)
    #func.example_plot(train_loader=train_loader,
    #                  dataset=dataset)
elif dataset == "FashionMNIST":
    train_loader, valid_loader, test_loader = func.FashionMNIST_make_loaders(batch_size=256)
    #func.example_plot(train_loader=train_loader,
    #                  dataset=dataset)
elif dataset == "CATDOG":
    train_loader, valid_loader, test_loader, class_names = func.CATDOG_make_loaders(batch_size=256)
    #NEEDS FIXING
    func.example_plot(train_loader=train_loader,
                      dataset=dataset)

if os.path.isfile(os.path.join(current_path, "SavedModels/", savedmodelname)):
    print("There is a saved model.")
    print("Loading the saved model...")
    model = func.load_model(savedmodelname)
else:
    if model == 1:
        model = m.Conv_NN_small(kernelsize=kernel, classes=classes)
    elif model == 2:
        model = m.Conv_NN_medium(kernelsize=kernel, classes=classes)
    elif model == 3:
        model = m.FNO_NN_small(kernelsize=kernel, classes=classes)
    elif model == 4:
        model = m.FNO_NN_medium(kernelsize=kernel, classes=classes)

    print(f"Training the {savedmodelname} model on the {dataset} dataset")

    model, train_losses, valid_losses = func.train_model(model=model,
                                                         train_loader=train_loader,
                                                         valid_loader=valid_loader,
                                                         patience=patience,
                                                         n_epochs=200)
    func.save_model(model, savedmodelname, patience, kernel)
    func.loss_plot(train_loss=train_losses, 
                   valid_loss=valid_losses, 
                   filename=filename,
                   patience=patience,
                   kernel=kernel)

func.count_parameters(model)

func.evaluate_model(model=model, 
                    test_loader=test_loader, 
                    batch_size=256)

func.sample_test(model=model, 
                 test_loader=test_loader,
                 filename=filename,
                 patience=patience,
                 kernel=kernel)
