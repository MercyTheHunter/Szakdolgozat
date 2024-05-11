import functions as func
import os

#Base parameters
patiences = {3,5,7} #3 or 5 or 7 training parameter (also used for save location)
kernels = {3,5,7} #3 or 5 or 7 model parameter (also used for save location)
models = {1,2,4,5} #1:CNN_small, 2:CNN_medium, 3:CNN_big, 4:FNN_small, 5:FNN_medium, 6:FNN_big
datasets = {1,2,3} #1: MNIST, 2: FashionMNIST, 3: CATDOG

mode = 1 #Training: 1, Testing: 2

current_path = os.path.dirname(os.path.realpath(__file__))

for data in datasets:

    #Setting base data parameters
    dataset, classes, in_channels = func.set_data_params(data)

    #Loading the dataset + Plot of example images
    train_loader, valid_loader, test_loader, class_names = func.make_loaders(batch_size=128, 
                                                                             dataset=dataset)
    func.example_plot(train_loader=train_loader,
                      dataset=dataset)

    for modelnum in models:
        #Set the current model name for saving
        savedmodelname, filename = func.set_model_name(model=modelnum,
                                                       dataset=dataset)
        for patience in patiences:
            for kernel in kernels:
                #Loading a trained model or training a new one
                if mode == 2:
                    print("Testing saved models...")
                    print("Loading the saved model...")
                    model = func.load_model(model_name=savedmodelname,
                                            kernel=kernel,
                                            patience=patience)
                else:
                    print("Training new models...")
                    model = func.set_model(modelnum=modelnum,
                                           kernel=kernel,
                                           classes=classes,
                                           in_channels=in_channels)

                    print(f"Training the {savedmodelname} model on the {dataset} dataset")

                    model = func.train_model(model=model,
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
                
                #Plot a confusion matrix of the model on the test images
                func.make_confusion_matrix(model=model,
                                           test_loader=test_loader,
                                           class_names=class_names,
                                           filename=filename,
                                           patience=patience,
                                           kernel=kernel,
                                           dataset=dataset)
                
func.user_test("11055.jpg", class_names)