import models as m
import functions as func
import os

#Base parameters
patience = 3 #3 or 5 or 7 training parameter (also used for save location)
kernel = 3 #3 or 5 or 7 model parameter (also used for save location)
model = 1 #1:CNN_small, 2:CNN_medium, 3:CNN_big, 4:FNN_small, 5:FNN_medium, 6:FNN_big

mode = 1 #Training: 1, Testing: 2

current_path = os.path.dirname(os.path.realpath(__file__))

dataset = "MNIST"
#dataset = "FashionMNIST"
#dataset = "CATDOG"

if model == 1:
    filename = dataset + "_CNN_small"
elif model == 2:
    filename = dataset + "_CNN_medium"
elif model == 3:
    filename = dataset + "_CNN_big"
elif model == 4:
    filename = dataset + "_FNN_small"
elif model == 5:
    filename = dataset + "_FNN_medium"
elif model == 6:
    filename = dataset + "_FNN_big"

savedmodelname = filename + ".pkl"

#Loading the dataset + Plot of example images
if dataset == "MNIST":
    classes = 10
    im_size = 28
    in_channels = 1
    train_loader, valid_loader, test_loader = func.MNIST_make_loaders(batch_size=128)
    #func.example_plot(train_loader=train_loader,
    #                  dataset=dataset)
elif dataset == "FashionMNIST":
    classes = 10
    im_size = 28
    in_channels = 1
    train_loader, valid_loader, test_loader = func.FashionMNIST_make_loaders(batch_size=128)
    #func.example_plot(train_loader=train_loader,
    #                  dataset=dataset)
elif dataset == "CATDOG":
    classes = 2
    im_size = 224
    in_channels = 3
    train_loader, valid_loader, test_loader, class_names = func.CATDOG_make_loaders(batch_size=128)
    #func.example_plot(train_loader=train_loader,
    #                  dataset=dataset)

#Loading a trained model or training a new one
if os.path.isfile(os.path.join(current_path, "SavedModels/", savedmodelname)):
    print("There is a saved model.")
    print("Loading the saved model...")
    model = func.load_model(model_name=savedmodelname)
else:
    if model == 1:
        model = m.Conv_NN_small(kernelsize=kernel, 
                                classes=classes, 
                                im_size=im_size, 
                                in_channels=in_channels)
    elif model == 2:
        model = m.Conv_NN_medium(kernelsize=kernel, 
                                 classes=classes, 
                                 im_size=im_size, 
                                 in_channels=in_channels)
    elif model == 3:
        model = m.Conv_NN_big(kernelsize=kernel, 
                              classes=classes, 
                              im_size=im_size, 
                              in_channels=in_channels)
    elif model == 4:
        model = m.FNO_NN_small(kernelsize=kernel, 
                               classes=classes, 
                               im_size=im_size, 
                               in_channels=in_channels)
    elif model == 5:
        model = m.FNO_NN_medium(kernelsize=kernel, 
                                classes=classes, 
                                im_size=im_size, 
                                in_channels=in_channels)
    elif model == 6:
        model = m.FNO_NN_big(kernelsize=kernel, 
                             classes=classes, 
                             im_size=im_size, 
                             in_channels=in_channels)

    print(f"Training the {savedmodelname} model on the {dataset} dataset")

    model, train_losses, valid_losses = func.train_model(model=model,
                                                         train_loader=train_loader,
                                                         valid_loader=valid_loader,
                                                         patience=patience,
                                                         n_epochs=200)
    func.save_model(model=model, 
                    model_name=savedmodelname, 
                    patience=patience, 
                    kernel=kernel)
    func.loss_plot(train_loss=train_losses, 
                   valid_loss=valid_losses, 
                   filename=filename,
                   patience=patience,
                   kernel=kernel)

#Count the parameters of the loaded or the newly trained model
func.count_parameters(model=model)

#Evaluate the loaded or the newly trained model
func.evaluate_model(model=model, 
                    test_loader=test_loader, 
                    batch_size=128,
                    classes=classes)

#Plot of a sample test from the images
func.sample_test(model=model, 
                 test_loader=test_loader,
                 filename=filename,
                 patience=patience,
                 kernel=kernel,
                 dataset=dataset)
