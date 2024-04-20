import models as m
import functions as func
import os

#Base parameters
patience = 5 #3 or 5 or 7 training parameter (also used for save location)
kernel = 7 #3 or 5 or 7 model parameter (also used for save location)
model = 1 #1:CNN_small, 2:CNN_medium, 3:FNN_small, 4:FNN_medium
classes = 10 #MNIST: 10, FashionMNIST: 10, CATDOG: 2
current_path = os.path.dirname(os.path.realpath(__file__))

dataset = "MNIST"
#dataset = "FashionMNIST"
#dataset = "CATDOG"

if model == 1:
    filename = dataset + "_CNN_small"
elif model == 2:
    filename = dataset + "_CNN_medium"
elif model == 3:
    filename = dataset + "_FNN_small"
elif model == 4:
    filename = dataset + "_FNN_medium"

savedmodelname = filename + ".pkl"

#Loading the dataset + Plot of example images
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
    func.CATDOG_example_plot(train_loader=train_loader,
                      dataset=dataset)

#Loading a trained model or training a new one
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

#Count the parameters of the loaded or the newly trained model
func.count_parameters(model)

#Evaluate the loaded or the newly trained model
func.evaluate_model(model=model, 
                    test_loader=test_loader, 
                    batch_size=256)

#Plot of a sample test from the images
func.sample_test(model=model, 
                 test_loader=test_loader,
                 filename=filename,
                 patience=patience,
                 kernel=kernel)
