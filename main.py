import models as m
import functions as func
import os

dataset = "MNIST"
#dataset = "FashionMNIST"
#dataset = "STL10"

#filename = dataset + "_CNN_small"
#filename = dataset + "_CNN_medium"
#filename = dataset + "_FNN_small"
filename = dataset + "_FNN_medium"

#savedmodelname = dataset + "_CNN_small.pkl"
#savedmodelname = dataset + "_CNN_medium.pkl"
#savedmodelname = dataset + "_FNN_small.pkl"
savedmodelname = dataset + "_FNN_medium.pkl"

current_path = os.path.dirname(os.path.realpath(__file__))
patience = 3 #3 or 5 or 7 (for save location)
kernel = 3 #3 or 5 or 7 (for save location)

if dataset == "MNIST":
    train_loader, valid_loader, test_loader = func.MNIST_make_loaders(batch_size=256)
    #func.example_plot(train_loader=train_loader,
    #                  dataset=dataset)
elif dataset == "FashionMNIST":
    train_loader, valid_loader, test_loader = func.FashionMNIST_make_loaders(batch_size=256)
    #func.example_plot(train_loader=train_loader,
    #                  dataset=dataset)
elif dataset == "STL10":
    train_loader, valid_loader, test_loader = func.STL10_make_loaders(batch_size=256)
    #NEEDS FIXING
    func.example_plot(train_loader=train_loader,
                      dataset=dataset)

if os.path.isfile(os.path.join(current_path, "SavedModels/", savedmodelname)):
    print("There is a saved model.")
    print("Loading the saved model...")
    model = func.load_model(savedmodelname)
else:
    #model = m.Conv_NN_small()
    #model = m.Conv_NN_medium()
    #model = m.FNO_NN_small()
    model = m.FNO_NN_medium()

    print(f"Training the {savedmodelname} model on the {dataset} dataset")

    model, train_losses, valid_losses = func.train_model(model=model,
                                                         train_loader=train_loader,
                                                         valid_loader=valid_loader,
                                                         patience=patience, #3 or 5 or 7
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
