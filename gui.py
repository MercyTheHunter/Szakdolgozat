import tkinter as tk
import customtkinter
from PIL import Image
import os

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    width = 1920
    height = 1080

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry(f"{self.width}x{self.height}")
        self.title("Model Evaluation Information")

        # configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure((0,1), weight=1)

        # set current path for images
        current_path = os.path.dirname(os.path.realpath(__file__))

        #####################################################################
        ### Sidebar
        #####################################################################

        self.sidebar_frame = customtkinter.CTkFrame(self, width=100, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Model Information", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="The Datasets", command=self.tab_event_1)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Model Scores", command=self.tab_event_2)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Small Models", command=self.tab_event_3)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Medium Models", command=self.tab_event_4)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Large Models", command=self.tab_event_5)
        self.sidebar_button_5.grid(row=5, column=0, padx=20, pady=10)
        self.sidebar_button_6 = customtkinter.CTkButton(self.sidebar_frame, text="Test", command=self.tab_event_6)
        self.sidebar_button_6.grid(row=6, column=0, padx=20, pady=10)
        
        #Appearance and UI scaling
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(30, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=10, column=0, padx=20, pady=(20, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["70%", "80%", "90%", "100%", "110%", "120%", "130%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 10))
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

        #####################################################################
        ### Tab frame1 - The datasets
        #####################################################################

        self.tab_frame_1 = customtkinter.CTkFrame(self,corner_radius=0, height=self.height)
        self.tab_frame_1.grid_columnconfigure(0, weight=1)
        self.tab_frame_1.grid_rowconfigure(0, weight=1)
        self.tabview_1 = customtkinter.CTkTabview(self.tab_frame_1, height=self.height)
        self.tabview_1.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        self.tabview_1.add("MNIST")
        self.tabview_1.tab("MNIST").grid_columnconfigure(0, weight=1)
        self.tabview_1.tab("MNIST").grid_rowconfigure(0, weight=1)
        
        self.label = customtkinter.CTkLabel(self.tabview_1.tab("MNIST"), font=("Liberation Seriff",20),
                     text="The MNIST (Modified National Institute of Standards and Technology) dataset is " +
                            "a large database containing handwritten digits from 0 to 9." + 
                            "\nThese digits are black and white images that have been normalized to be 28x28 pixels." +
                            "\nThe dataset is commonly used for testing various models in machine learning." +
                            "\nThe dataset contains 60000 training images and 10000 testing images" +
                            "\nBelow we can see an example of these images")        
        
        self.label.grid(row=0, column=0)
        self.image_t1_ep_M = customtkinter.CTkImage(Image.open(os.path.join(current_path,"MNIST_example_plot.png")), size=(1280, 150))
        self.image_t1_ep_M_label = customtkinter.CTkLabel(self.tabview_1.tab("MNIST"), text ="", image=self.image_t1_ep_M)
        self.image_t1_ep_M_label.grid(row=1, column=0)

        self.b_padding_1 = customtkinter.CTkLabel(self.tabview_1.tab("MNIST"), font=("Liberation Seriff",20), text="")
        self.b_padding_1.grid(row=3, column=0)
        self.b_padding_2 = customtkinter.CTkLabel(self.tabview_1.tab("MNIST"), font=("Liberation Seriff",20), text="")
        self.b_padding_2.grid(row=4, column=0)
        self.b_padding_3 = customtkinter.CTkLabel(self.tabview_1.tab("MNIST"), font=("Liberation Seriff",20), text="")
        self.b_padding_3.grid(row=5, column=0)
        
        self.tabview_1.add("FashionMNIST")
        self.tabview_1.tab("FashionMNIST").grid_columnconfigure(0, weight=1)
        self.tabview_1.tab("FashionMNIST").grid_rowconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self.tabview_1.tab("FashionMNIST"), font=("Liberation Seriff",20),
                     text="The Fashion MNIST dataset is similar to the MNIST dataset, it is also" +
                            "a large database containing different clothes ranging from trouserts to bags." +
                            "\nThe dataset was designed to be the successor to the MNIST dataset, and similarly it's used in machine learning."
                            "\nThese images are also black and white images that have been normalized to be 28x28 pixels." + 
                            "\nThe dataset contains 60000 training images and 10000 testing images" +
                            "\nBelow we can see an example of these images")        
        self.label.grid(row=0, column=0)

        self.image_t1_ep_F = customtkinter.CTkImage(Image.open(os.path.join(current_path, "FashionMNIST_example_plot.png")), size=(1280, 150))
        self.image_t1_ep_F_label = customtkinter.CTkLabel(self.tabview_1.tab("FashionMNIST"), text ="", image=self.image_t1_ep_F)
        self.image_t1_ep_F_label.grid(row=1, column=0)

        self.b_padding_1 = customtkinter.CTkLabel(self.tabview_1.tab("FashionMNIST"), font=("Liberation Seriff",20), text="")
        self.b_padding_1.grid(row=3, column=0)
        self.b_padding_2 = customtkinter.CTkLabel(self.tabview_1.tab("FashionMNIST"), font=("Liberation Seriff",20), text="")
        self.b_padding_2.grid(row=4, column=0)
        self.b_padding_3 = customtkinter.CTkLabel(self.tabview_1.tab("FashionMNIST"), font=("Liberation Seriff",20), text="")
        self.b_padding_3.grid(row=5, column=0)
        
        self.tabview_1.add("CATDOG")
        self.tabview_1.tab("CATDOG").grid_columnconfigure(0, weight=1)
        self.tabview_1.tab("CATDOG").grid_rowconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self.tabview_1.tab("CATDOG"), font=("Liberation Seriff",20),
                     text="The CATDOG dataset is a large database containing colorful pictures of cats and dogs." +
                            "\nThe dataset was inspired by the famous kaggle competiton 10 years ago," +
                            "\nwhere participants had to make an artificial intelligence which could decide if the given image was a cat or a dog." +
                            "\nThese images are colored images and come in many shapes and sizes." + 
                            "\nThe dataset contains around 19000 training images and around 6000 testing images" +
                            "\nAbove we can see an example of these images")        
        self.label.grid(row=0, column=0)

        self.image_t1_ep_C = customtkinter.CTkImage(Image.open(os.path.join(current_path, "CATDOG_example_plot.png")), size=(1280, 150))
        self.image_t1_ep_C_label = customtkinter.CTkLabel(self.tabview_1.tab("CATDOG"), text ="", image=self.image_t1_ep_C)
        self.image_t1_ep_C_label.grid(row=1, column=0)

        self.b_padding_1 = customtkinter.CTkLabel(self.tabview_1.tab("CATDOG"), font=("Liberation Seriff",20), text="")
        self.b_padding_1.grid(row=3, column=0)
        self.b_padding_2 = customtkinter.CTkLabel(self.tabview_1.tab("CATDOG"), font=("Liberation Seriff",20), text="")
        self.b_padding_2.grid(row=4, column=0)
        self.b_padding_3 = customtkinter.CTkLabel(self.tabview_1.tab("CATDOG"), font=("Liberation Seriff",20), text="")
        self.b_padding_3.grid(row=5, column=0)

        #####################################################################
        ### Tab frame2 - Model scores
        #####################################################################
        self.tab_frame_2 = customtkinter.CTkFrame(self,corner_radius=0, height=self.height)
        self.tab_frame_2.grid_columnconfigure(0, weight=1)
        self.tab_frame_2.grid_rowconfigure(0, weight=1)
        self.tabview_2 = customtkinter.CTkTabview(self.tab_frame_2, height=self.height)
        self.tabview_2.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")

        self.tabview_2.add("Patience 3")
        self.tabview_2.tab("Patience 3").grid_columnconfigure(0, weight=1)
        self.tabview_2.tab("Patience 3").grid_rowconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self.tabview_2.tab("Patience 3"), font=("Liberation Seriff",20),
                     text="Overall Model scores with the Patience number being equal to 3")        
        self.label.grid(row=0, column=0)

        self.image_t2_p3 = customtkinter.CTkImage(Image.open(os.path.join(current_path, "ModelScores_P3.png")), size=(800, 600))
        self.image_t2_p3_label = customtkinter.CTkLabel(self.tabview_2.tab("Patience 3"), text ="", image=self.image_t2_p3)
        self.image_t2_p3_label.grid(row=1, column=0)

        self.b_padding_1 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 3"), font=("Liberation Seriff",20), text="")
        self.b_padding_1.grid(row=3, column=0)
        self.b_padding_2 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 3"), font=("Liberation Seriff",20), text="")
        self.b_padding_2.grid(row=4, column=0)
        self.b_padding_3 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 3"), font=("Liberation Seriff",20), text="")
        self.b_padding_3.grid(row=5, column=0)

        self.tabview_2.add("Patience 5")
        self.tabview_2.tab("Patience 5").grid_columnconfigure(0, weight=1)
        self.tabview_2.tab("Patience 5").grid_rowconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self.tabview_2.tab("Patience 5"), font=("Liberation Seriff",20),
                     text="Overall Model scores with the Patience number being equal to 5")        
        self.label.grid(row=0, column=0)

        self.image_t2_p5 = customtkinter.CTkImage(Image.open(os.path.join(current_path, "ModelScores_P5.png")), size=(800, 600))
        self.image_t2_p5_label = customtkinter.CTkLabel(self.tabview_2.tab("Patience 5"), text ="", image=self.image_t2_p5)
        self.image_t2_p5_label.grid(row=1, column=0)

        self.b_padding_1 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 5"), font=("Liberation Seriff",20), text="")
        self.b_padding_1.grid(row=3, column=0)
        self.b_padding_2 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 5"), font=("Liberation Seriff",20), text="")
        self.b_padding_2.grid(row=4, column=0)
        self.b_padding_3 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 5"), font=("Liberation Seriff",20), text="")
        self.b_padding_3.grid(row=5, column=0)

        self.tabview_2.add("Patience 7")
        self.tabview_2.tab("Patience 7").grid_columnconfigure(0, weight=1)
        self.tabview_2.tab("Patience 7").grid_rowconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self.tabview_2.tab("Patience 7"), font=("Liberation Seriff",20),
                     text="Overall Model scores with the Patience number being equal to 7")        
        self.label.grid(row=0, column=0)

        self.image_t2_p7 = customtkinter.CTkImage(Image.open(os.path.join(current_path, "ModelScores_P7.png")), size=(800, 600))
        self.image_t2_p7_label = customtkinter.CTkLabel(self.tabview_2.tab("Patience 7"), text ="", image=self.image_t2_p7)
        self.image_t2_p7_label.grid(row=1, column=0)

        self.b_padding_1 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 7"), font=("Liberation Seriff",20), text="")
        self.b_padding_1.grid(row=3, column=0)
        self.b_padding_2 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 7"), font=("Liberation Seriff",20), text="")
        self.b_padding_2.grid(row=4, column=0)
        self.b_padding_3 = customtkinter.CTkLabel(self.tabview_2.tab("Patience 7"), font=("Liberation Seriff",20), text="")
        self.b_padding_3.grid(row=5, column=0)

        #####################################################################
        ### Tab frame3 - The small models
        #####################################################################
        self.tab_frame_3 = customtkinter.CTkScrollableFrame(self,corner_radius=0, height=self.height)
        self.tab_frame_3.grid_columnconfigure(0, weight=1)
        self.tab_frame_3.grid_rowconfigure(0, weight=1)
        self.tabview_3 = customtkinter.CTkTabview(self.tab_frame_3, height=self.height)
        self.tabview_3.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        self.setup_small_model_tab_view(self.tabview_3, current_path,3,3)
        self.setup_small_model_tab_view(self.tabview_3, current_path,3,5)
        self.setup_small_model_tab_view(self.tabview_3, current_path,3,7)
        self.setup_small_model_tab_view(self.tabview_3, current_path,5,3)
        self.setup_small_model_tab_view(self.tabview_3, current_path,5,5)
        self.setup_small_model_tab_view(self.tabview_3, current_path,5,7)
        self.setup_small_model_tab_view(self.tabview_3, current_path,7,3)
        self.setup_small_model_tab_view(self.tabview_3, current_path,7,5)
        self.setup_small_model_tab_view(self.tabview_3, current_path,7,7)

        #####################################################################
        ### Tab frame4 - The medium models
        #####################################################################
        self.tab_frame_4 = customtkinter.CTkScrollableFrame(self,corner_radius=0, height=self.height)
        self.tab_frame_4.grid_columnconfigure(0, weight=1)
        self.tab_frame_4.grid_rowconfigure(0, weight=1)
        self.tabview_4 = customtkinter.CTkTabview(self.tab_frame_4, height=self.height)
        self.tabview_4.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")

        self.setup_medium_model_tab_view(self.tabview_4, current_path,3,3)
        self.setup_medium_model_tab_view(self.tabview_4, current_path,3,5)
        self.setup_medium_model_tab_view(self.tabview_4, current_path,3,7)
        self.setup_medium_model_tab_view(self.tabview_4, current_path,5,3)
        self.setup_medium_model_tab_view(self.tabview_4, current_path,5,5)
        self.setup_medium_model_tab_view(self.tabview_4, current_path,5,7)
        self.setup_medium_model_tab_view(self.tabview_4, current_path,7,3)
        self.setup_medium_model_tab_view(self.tabview_4, current_path,7,5)
        self.setup_medium_model_tab_view(self.tabview_4, current_path,7,7)

        #####################################################################
        ### Tab frame5 - The large models
        #####################################################################
        self.tab_frame_5 = customtkinter.CTkScrollableFrame(self,corner_radius=0, height=self.height)
        self.tab_frame_5.grid_columnconfigure(0, weight=1)
        self.tab_frame_5.grid_rowconfigure(0, weight=1)
        self.tabview_5 = customtkinter.CTkTabview(self.tab_frame_5, height=self.height)
        self.tabview_5.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")

        self.setup_large_model_tab_view(self.tabview_5, current_path,3,3)
        self.setup_large_model_tab_view(self.tabview_5, current_path,3,5)
        self.setup_large_model_tab_view(self.tabview_5, current_path,3,7)
        self.setup_large_model_tab_view(self.tabview_5, current_path,5,3)
        self.setup_large_model_tab_view(self.tabview_5, current_path,5,5)
        self.setup_large_model_tab_view(self.tabview_5, current_path,5,7)
        self.setup_large_model_tab_view(self.tabview_5, current_path,7,3)
        self.setup_large_model_tab_view(self.tabview_5, current_path,7,5)
        self.setup_large_model_tab_view(self.tabview_5, current_path,7,7)

        #####################################################################
        ### Tab frame6 - The image test
        #####################################################################
        self.tab_frame_6 = customtkinter.CTkFrame(self,corner_radius=0, height=self.height)
        self.tab_frame_6.grid_columnconfigure(0, weight=1)
        self.tab_frame_6.grid_rowconfigure(0, weight=1)

        self.tabview_6 = customtkinter.CTkTabview(self.tab_frame_6, height=self.height)
        self.tabview_6.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")

        self.tabview_6.add("Image test")
        self.tabview_6.tab("Image test").grid_columnconfigure(0, weight=1)
        self.tabview_6.tab("Image test").grid_rowconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self.tabview_6.tab("Image test"), font=("Liberation Seriff",20),
                     text="Chosen image for testing one of the models")        
        self.label.grid(row=0, column=0)

        self.image = customtkinter.CTkImage(Image.open(os.path.join(current_path, "UserTest/11055.jpg")), size=(800, 600))
        self.image_label = customtkinter.CTkLabel(self.tabview_6.tab("Image test"), text ="", image=self.image)
        self.image_label.grid(row=1, column=0)

        self.b_padding_1 = customtkinter.CTkLabel(self.tabview_6.tab("Image test"), font=("Liberation Seriff",20), text="")
        self.b_padding_1.grid(row=3, column=0)
        self.b_padding_2 = customtkinter.CTkLabel(self.tabview_6.tab("Image test"), font=("Liberation Seriff",20), text="")
        self.b_padding_2.grid(row=4, column=0)
        self.b_padding_3 = customtkinter.CTkLabel(self.tabview_6.tab("Image test"), font=("Liberation Seriff",20), text="")
        self.b_padding_3.grid(row=5, column=0)

        

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
    def tab_event_1(self):
        self.tab_frame_2.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_5.grid_forget()
        self.tab_frame_6.grid_forget()
        self.tab_frame_1.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
    def tab_event_2(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_5.grid_forget()
        self.tab_frame_6.grid_forget()
        self.tab_frame_2.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
    def tab_event_3(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_2.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_5.grid_forget()
        self.tab_frame_6.grid_forget()
        self.tab_frame_3.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
    def tab_event_4(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_2.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_5.grid_forget()
        self.tab_frame_6.grid_forget()
        self.tab_frame_4.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
    def tab_event_5(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_2.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_6.grid_forget()
        self.tab_frame_5.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
    def tab_event_6(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_2.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_5.grid_forget()
        self.tab_frame_6.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")

    def mousewheel(self, event):
        global count
        #Differentiate Windows or Linux wheel events and respond to them
        if event.num == 5 or event.delta == -120:
            count -= 1
        if event.num == 4 or event.delta == 120:
            count += 1

    def get_plot_path(self,current_path, patience, kernel):
        current_path = os.path.join(current_path, "TestPlots/")
        patience = "Patience" + str(patience) +"/"
        patience_folder = os.path.join(current_path, patience)
        kernel = "Kernel" + str(kernel) + "/"
        kernel_folder = os.path.join(patience_folder, kernel)
        path = kernel_folder
        return path
    
    def setup_small_model_tab_view(self, tabview, current_path, patience, kernel):
        tabname = "Patience: " + str(patience) + " Kernel: " + str(kernel)
        tabview.add(tabname)
        tabview.tab(tabname).grid_columnconfigure(0, weight=1)
        tabview.tab(tabname).grid_rowconfigure(0, weight=1)
        path = self.get_plot_path(current_path,patience,kernel)
        #Dataset1 - MNIST
        self.dataset1_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20),
                                                     text="The Following models were trained on the MNIST Dataset \n The Convolutional Model (CNN - Left) VS The Fourier Neural Operator Model (FNN - Right)")
        self.dataset1_label.grid(row=0, column=0, columnspan=2)
        #Learning Rate Plot
        self.image_cnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_CNN_small_loss_plot.png")), size=(800, 600))
        self.image_cnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the CNN Model", compound="top", image=self.image_cnn_lr)
        self.image_cnn_lr_label.grid(row=1, column=0)

        self.image_fnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_FNN_small_loss_plot.png")), size=(800, 600))
        self.image_fnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the FNN Model", compound="top", image=self.image_fnn_lr)
        self.image_fnn_lr_label.grid(row=1, column=1)
        self.padding1 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding1.grid(row=2, column=0, columnspan=2)
        #Sample Test Plot
        self.image_cnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_CNN_small_sample_test_plot.png")), size=(800, 250))
        self.image_cnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the CNN Model", compound="top", image=self.image_cnn_st)
        self.image_cnn_st_label.grid(row=3, column=0)

        self.image_fnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_FNN_small_sample_test_plot.png")), size=(800, 250))
        self.image_fnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the FNN Model", compound="top", image=self.image_fnn_st)
        self.image_fnn_st_label.grid(row=3, column=1)
        self.padding2 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding2.grid(row=4, column=0, columnspan=2)
        #Confusion Matrix
        self.image_cnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_CNN_small_confusion_matrix_plot.png")), size=(800, 600))
        self.image_cnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the CNN Model", compound="top", image=self.image_cnn_cm)
        self.image_cnn_cm_label.grid(row=5, column=0)

        self.image_fnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_FNN_small_confusion_matrix_plot.png")), size=(800, 600))
        self.image_fnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the FNN Model", compound="top", image=self.image_fnn_cm)
        self.image_fnn_cm_label.grid(row=5, column=1)
        #Dataset2 - FashionMNIST
        self.dataset2_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20),
                                                     text="The Following models were trained on the FashionMNIST Dataset \n The Convolutional Model (CNN - Left) VS The Fourier Neural Operator Model (FNN - Right)")
        self.dataset2_label.grid(row=6, column=0, columnspan=2)
        #Learning Rate Plot
        self.image_cnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_CNN_small_loss_plot.png")), size=(800, 600))
        self.image_cnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the CNN Model", compound="top", image=self.image_cnn_lr)
        self.image_cnn_lr_label.grid(row=7, column=0)

        self.image_fnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_FNN_small_loss_plot.png")), size=(800, 600))
        self.image_fnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the FNN Model", compound="top", image=self.image_fnn_lr)
        self.image_fnn_lr_label.grid(row=7, column=1)
        self.padding1 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding1.grid(row=8, column=0, columnspan=2)
        #Sample Test Plot
        self.image_cnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_CNN_small_sample_test_plot.png")), size=(800, 250))
        self.image_cnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the CNN Model", compound="top", image=self.image_cnn_st)
        self.image_cnn_st_label.grid(row=9, column=0)

        self.image_fnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_FNN_small_sample_test_plot.png")), size=(800, 250))
        self.image_fnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the FNN Model", compound="top", image=self.image_fnn_st)
        self.image_fnn_st_label.grid(row=9, column=1)
        self.padding2 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding2.grid(row=10, column=0, columnspan=2)
        #Confusion Matrix
        self.image_cnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_CNN_small_confusion_matrix_plot.png")), size=(800, 600))
        self.image_cnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the CNN Model", compound="top", image=self.image_cnn_cm)
        self.image_cnn_cm_label.grid(row=11, column=0)

        self.image_fnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_FNN_small_confusion_matrix_plot.png")), size=(800, 600))
        self.image_fnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the FNN Model", compound="top", image=self.image_fnn_cm)
        self.image_fnn_cm_label.grid(row=11, column=1)
        #Dataset3 - CATDOG
        self.dataset3_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20),
                                                     text="The Following models were trained on the CATDOG Dataset \n The Convolutional Model (CNN - Left) VS The Fourier Neural Operator Model (FNN - Right)")
        self.dataset3_label.grid(row=12, column=0, columnspan=2)
        #Learning Rate Plot
        self.image_cnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_small_loss_plot.png")), size=(800, 600))
        self.image_cnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the CNN Model", compound="top", image=self.image_cnn_lr)
        self.image_cnn_lr_label.grid(row=13, column=0)

        self.image_fnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_small_loss_plot.png")), size=(800, 600))
        self.image_fnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the FNN Model", compound="top", image=self.image_fnn_lr)
        self.image_fnn_lr_label.grid(row=13, column=1)
        self.padding1 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding1.grid(row=14, column=0, columnspan=2)
        #Sample Test Plot
        self.image_cnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_small_sample_test_plot.png")), size=(800, 250))
        self.image_cnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the CNN Model", compound="top", image=self.image_cnn_st)
        self.image_cnn_st_label.grid(row=15, column=0)

        self.image_fnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_small_sample_test_plot.png")), size=(800, 250))
        self.image_fnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the FNN Model", compound="top", image=self.image_fnn_st)
        self.image_fnn_st_label.grid(row=15, column=1)
        self.padding2 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding2.grid(row=16, column=0, columnspan=2)
        #Confusion Matrix
        self.image_cnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_small_confusion_matrix_plot.png")), size=(800, 600))
        self.image_cnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the CNN Model", compound="top", image=self.image_cnn_cm)
        self.image_cnn_cm_label.grid(row=17, column=0)

        self.image_fnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_small_confusion_matrix_plot.png")), size=(800, 600))
        self.image_fnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the FNN Model", compound="top", image=self.image_fnn_cm)
        self.image_fnn_cm_label.grid(row=17, column=1)
    
    def setup_medium_model_tab_view(self, tabview, current_path, patience, kernel):
        tabname = "Patience: " + str(patience) + " Kernel: " + str(kernel)
        tabview.add(tabname)
        tabview.tab(tabname).grid_columnconfigure(0, weight=1)
        tabview.tab(tabname).grid_rowconfigure(0, weight=1)
        path = self.get_plot_path(current_path,patience,kernel)
        #Dataset1 - MNIST
        self.dataset1_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20),
                                                     text="The Following models were trained on the MNIST Dataset \n The Convolutional Model (CNN - Left) VS The Fourier Neural Operator Model (FNN - Right)")
        self.dataset1_label.grid(row=0, column=0, columnspan=2)
        #Learning Rate Plot
        self.image_cnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_CNN_medium_loss_plot.png")), size=(800, 600))
        self.image_cnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the CNN Model", compound="top", image=self.image_cnn_lr)
        self.image_cnn_lr_label.grid(row=1, column=0)

        self.image_fnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_FNN_medium_loss_plot.png")), size=(800, 600))
        self.image_fnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the FNN Model", compound="top", image=self.image_fnn_lr)
        self.image_fnn_lr_label.grid(row=1, column=1)
        self.padding1 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding1.grid(row=2, column=0, columnspan=2)
        #Sample Test Plot
        self.image_cnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_CNN_medium_sample_test_plot.png")), size=(800, 250))
        self.image_cnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the CNN Model", compound="top", image=self.image_cnn_st)
        self.image_cnn_st_label.grid(row=3, column=0)

        self.image_fnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_FNN_medium_sample_test_plot.png")), size=(800, 250))
        self.image_fnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the FNN Model", compound="top", image=self.image_fnn_st)
        self.image_fnn_st_label.grid(row=3, column=1)
        self.padding2 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding2.grid(row=4, column=0, columnspan=2)
        #Confusion Matrix
        self.image_cnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_CNN_medium_confusion_matrix_plot.png")), size=(800, 600))
        self.image_cnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the CNN Model", compound="top", image=self.image_cnn_cm)
        self.image_cnn_cm_label.grid(row=5, column=0)

        self.image_fnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "MNIST_FNN_medium_confusion_matrix_plot.png")), size=(800, 600))
        self.image_fnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the FNN Model", compound="top", image=self.image_fnn_cm)
        self.image_fnn_cm_label.grid(row=5, column=1)
        #Dataset2 - FashionMNIST
        self.dataset2_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20),
                                                     text="The Following models were trained on the FashionMNIST Dataset \n The Convolutional Model (CNN - Left) VS The Fourier Neural Operator Model (FNN - Right)")
        self.dataset2_label.grid(row=6, column=0, columnspan=2)
        #Learning Rate Plot
        self.image_cnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_CNN_medium_loss_plot.png")), size=(800, 600))
        self.image_cnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the CNN Model", compound="top", image=self.image_cnn_lr)
        self.image_cnn_lr_label.grid(row=7, column=0)

        self.image_fnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_FNN_medium_loss_plot.png")), size=(800, 600))
        self.image_fnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the FNN Model", compound="top", image=self.image_fnn_lr)
        self.image_fnn_lr_label.grid(row=7, column=1)
        self.padding1 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding1.grid(row=8, column=0, columnspan=2)
        #Sample Test Plot
        self.image_cnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_CNN_medium_sample_test_plot.png")), size=(800, 250))
        self.image_cnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the CNN Model", compound="top", image=self.image_cnn_st)
        self.image_cnn_st_label.grid(row=9, column=0)

        self.image_fnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_FNN_medium_sample_test_plot.png")), size=(800, 250))
        self.image_fnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the FNN Model", compound="top", image=self.image_fnn_st)
        self.image_fnn_st_label.grid(row=9, column=1)
        self.padding2 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding2.grid(row=10, column=0, columnspan=2)
        #Confusion Matrix
        self.image_cnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_CNN_medium_confusion_matrix_plot.png")), size=(800, 600))
        self.image_cnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the CNN Model", compound="top", image=self.image_cnn_cm)
        self.image_cnn_cm_label.grid(row=11, column=0)

        self.image_fnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "FashionMNIST_FNN_medium_confusion_matrix_plot.png")), size=(800, 600))
        self.image_fnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the FNN Model", compound="top", image=self.image_fnn_cm)
        self.image_fnn_cm_label.grid(row=11, column=1)
        #Dataset3 - CATDOG
        self.dataset3_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20),
                                                     text="The Following models were trained on the CATDOG Dataset \n The Convolutional Model (CNN - Left) VS The Fourier Neural Operator Model (FNN - Right)")
        self.dataset3_label.grid(row=12, column=0, columnspan=2)
        #Learning Rate Plot
        self.image_cnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_medium_loss_plot.png")), size=(800, 600))
        self.image_cnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the CNN Model", compound="top", image=self.image_cnn_lr)
        self.image_cnn_lr_label.grid(row=13, column=0)

        self.image_fnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_medium_loss_plot.png")), size=(800, 600))
        self.image_fnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the FNN Model", compound="top", image=self.image_fnn_lr)
        self.image_fnn_lr_label.grid(row=13, column=1)
        self.padding1 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding1.grid(row=14, column=0, columnspan=2)
        #Sample Test Plot
        self.image_cnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_medium_sample_test_plot.png")), size=(800, 250))
        self.image_cnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the CNN Model", compound="top", image=self.image_cnn_st)
        self.image_cnn_st_label.grid(row=15, column=0)

        self.image_fnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_medium_sample_test_plot.png")), size=(800, 250))
        self.image_fnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the FNN Model", compound="top", image=self.image_fnn_st)
        self.image_fnn_st_label.grid(row=15, column=1)
        self.padding2 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding2.grid(row=16, column=0, columnspan=2)
        #Confusion Matrix
        self.image_cnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_medium_confusion_matrix_plot.png")), size=(800, 600))
        self.image_cnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the CNN Model", compound="top", image=self.image_cnn_cm)
        self.image_cnn_cm_label.grid(row=17, column=0)

        self.image_fnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_medium_confusion_matrix_plot.png")), size=(800, 600))
        self.image_fnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the FNN Model", compound="top", image=self.image_fnn_cm)
        self.image_fnn_cm_label.grid(row=17, column=1)
    
    def setup_large_model_tab_view(self, tabview, current_path, patience, kernel):
        tabname = "Patience: " + str(patience) + " Kernel: " + str(kernel)
        tabview.add(tabname)
        tabview.tab(tabname).grid_columnconfigure(0, weight=1)
        tabview.tab(tabname).grid_rowconfigure(0, weight=1)
        path = self.get_plot_path(current_path,patience,kernel)
        #Dataset1 - CATDOG
        self.dataset1_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20),
                                                     text="The Following models were trained on the CATDOG Dataset \n The Convolutional Model (CNN - Left) VS The Fourier Neural Operator Model (FNN - Right)")
        self.dataset1_label.grid(row=0, column=0, columnspan=2)
        #Learning Rate Plot
        self.image_cnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_big_loss_plot.png")), size=(800, 600))
        self.image_cnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the CNN Model", compound="top", image=self.image_cnn_lr)
        self.image_cnn_lr_label.grid(row=1, column=0)

        self.image_fnn_lr = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_big_loss_plot.png")), size=(800, 600))
        self.image_fnn_lr_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Learning Rate Graph of the FNN Model", compound="top", image=self.image_fnn_lr)
        self.image_fnn_lr_label.grid(row=1, column=1)
        self.padding1 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding1.grid(row=2, column=0, columnspan=2)
        #Sample Test Plot
        self.image_cnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_big_sample_test_plot.png")), size=(800, 250))
        self.image_cnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the CNN Model", compound="top", image=self.image_cnn_st)
        self.image_cnn_st_label.grid(row=3, column=0)

        self.image_fnn_st = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_big_sample_test_plot.png")), size=(800, 250))
        self.image_fnn_st_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Sample Test of the FNN Model", compound="top", image=self.image_fnn_st)
        self.image_fnn_st_label.grid(row=3, column=1)
        self.padding2 = customtkinter.CTkLabel(tabview.tab(tabname), text="")
        self.padding2.grid(row=4, column=0, columnspan=2)
        #Confusion Matrix
        self.image_cnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_CNN_big_confusion_matrix_plot.png")), size=(800, 600))
        self.image_cnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the CNN Model", compound="top", image=self.image_cnn_cm)
        self.image_cnn_cm_label.grid(row=5, column=0)

        self.image_fnn_cm = customtkinter.CTkImage(Image.open(os.path.join(path, "CATDOG_FNN_big_confusion_matrix_plot.png")), size=(800, 600))
        self.image_fnn_cm_label = customtkinter.CTkLabel(tabview.tab(tabname), font=("Liberation Seriff",20), fg_color="black",
                                                         text ="Confusion Matrix of the FNN Model", compound="top", image=self.image_fnn_cm)
        self.image_fnn_cm_label.grid(row=5, column=1)


 
if __name__ == "__main__":
    app = App()
    app.mainloop()