import tkinter as tk
import customtkinter
from PIL import Image
import os

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    width = 1280
    height = 720

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry(f"{self.width}x{self.height}")
        self.title("FNO.py")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # set current path for images
        current_path = os.path.dirname(os.path.realpath(__file__))

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=100, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="CustomTkinter", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Neural Operators", command=self.tab_event_1)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="The Datasets", command=self.tab_event_2)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Comparison", command=self.tab_event_3)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Test", command=self.tab_event_4)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(30, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=8, column=0, padx=20, pady=(20, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))

        #Tab frames
        self.tab_frame_1 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_1.grid_columnconfigure(0, weight=1)
        self.tab_frame_1.grid_rowconfigure(0, weight=1)
        self.tabview_1 = customtkinter.CTkTabview(self.tab_frame_1)
        self.tabview_1.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.tabview_1.add("Neural Network")
        self.tabview_1.add("Neural Operator")
        self.tabview_1.add("Fourier Neural Operator")

        self.tab_frame_2 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_2.grid_columnconfigure(0, weight=1)
        self.tab_frame_2.grid_rowconfigure(0, weight=1)
        self.tabview_2 = customtkinter.CTkTabview(self.tab_frame_2)
        self.tabview_2.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.tabview_2.add("MNIST")
        self.tabview_2.add("FashionMNIST")
        self.tabview_2.add("STL-10")

        self.tabview_2.tab("MNIST").grid_columnconfigure(0, weight=1)
        self.tabview_2.tab("MNIST").grid_rowconfigure(0, weight=1)
        self.image_t2_ep = customtkinter.CTkImage(Image.open(current_path + "/MNIST_example_plot.png"), size=(854, 480))
        self.image_t2_ep_label = customtkinter.CTkLabel(self.tabview_2.tab("MNIST"), text ="", image=self.image_t2_ep)
        self.image_t2_ep_label.grid(row=0, column=0)

        self.tabview_2.tab("FashionMNIST").grid_columnconfigure(0, weight=1)
        self.tabview_2.tab("FashionMNIST").grid_rowconfigure(0, weight=1)
        self.image_t2_ep = customtkinter.CTkImage(Image.open(current_path + "/FashionMNIST_example_plot.png"), size=(854, 480))
        self.image_t2_ep_label = customtkinter.CTkLabel(self.tabview_2.tab("FashionMNIST"), text ="", image=self.image_t2_ep)
        self.image_t2_ep_label.grid(row=0, column=0)

        self.tabview_2.tab("STL-10").grid_columnconfigure(0, weight=1)
        self.tabview_2.tab("STL-10").grid_rowconfigure(0, weight=1)
        self.image_t2_ep = customtkinter.CTkImage(Image.open(current_path + "/STL10_example_plot.png"), size=(854, 480))
        self.image_t2_ep_label = customtkinter.CTkLabel(self.tabview_2.tab("STL10"), text ="", image=self.image_t2_ep)
        self.image_t2_ep_label.grid(row=0, column=0)

        self.tab_frame_3 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_3.grid_columnconfigure(0, weight=1)
        self.tab_frame_3.grid_rowconfigure(0, weight=1)
        self.tabview_3 = customtkinter.CTkTabview(self.tab_frame_3)
        self.tabview_3.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.tabview_3.add("A Regular Neural Network")
        self.tabview_3.add("A Neural Network With The FNO")
        self.tabview_3.add("Learning Rate")
        self.tabview_3.add("Accuracy")

        self.tabview_3.tab("Learning Rate").grid_columnconfigure(0, weight=1)
        self.tabview_3.tab("Learning Rate").grid_rowconfigure(0, weight=1)
        self.image_t3_lr = customtkinter.CTkImage(Image.open(current_path + "/loss_plot.png"), size=(854, 480))
        self.image_t3_lr_label = customtkinter.CTkLabel(self.tabview_3.tab("Learning Rate"), text ="", image=self.image_t3_lr)
        self.image_t3_lr_label.grid(row=0, column=0)

        self.tab_frame_4 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_4.grid_columnconfigure(0, weight=1)
        self.tab_frame_4.grid_rowconfigure(0, weight=1)
        self.tabview_4 = customtkinter.CTkTabview(self.tab_frame_4)
        self.tabview_4.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.tabview_4.add("Placeholder")

        #Set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
    def tab_event_1(self):
        self.tab_frame_2.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_1.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
    def tab_event_2(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_2.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
    def tab_event_3(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_2.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_3.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
    def tab_event_4(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_2.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
 
if __name__ == "__main__":
    app = App()
    app.mainloop()