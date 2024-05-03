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

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

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
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Training parameters", command=self.tab_event_2)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Small Models", command=self.tab_event_3)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Medium Models", command=self.tab_event_4)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Big Models", command=self.tab_event_5)
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
        ### Tab frames
        #####################################################################

        #Tabframe 1 - The Datasets
        self.tab_frame_1 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_1.grid_columnconfigure(0, weight=1)
        self.tab_frame_1.grid_rowconfigure(0, weight=1)
        self.tabview_1 = customtkinter.CTkTabview(self.tab_frame_1)
        self.tabview_1.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        self.tabview_1.add("MNIST")
        self.tabview_1.tab("MNIST").grid_columnconfigure(0, weight=1)
        self.tabview_1.tab("MNIST").grid_rowconfigure(0, weight=1)
        self.image_t1_ep_M = customtkinter.CTkImage(Image.open(os.path.join(current_path,"MNIST_example_plot.png")), size=(1280, 150))
        self.image_t1_ep_M_label = customtkinter.CTkLabel(self.tabview_1.tab("MNIST"), text ="", image=self.image_t1_ep_M)
        self.image_t1_ep_M_label.grid(row=0, column=0)
        self.textbox = customtkinter.CTkTextbox(self.tabview_1.tab("MNIST"))
        self.textbox.grid(row=1, column=0, padx=(20, 0), pady=(20, 0), sticky="nsew")
        
        self.tabview_1.add("FashionMNIST")
        self.tabview_1.tab("FashionMNIST").grid_columnconfigure(0, weight=1)
        self.tabview_1.tab("FashionMNIST").grid_rowconfigure(0, weight=1)
        self.image_t1_ep_F = customtkinter.CTkImage(Image.open(os.path.join(current_path, "FashionMNIST_example_plot.png")), size=(1280, 150))
        self.image_t1_ep_F_label = customtkinter.CTkLabel(self.tabview_1.tab("FashionMNIST"), text ="", image=self.image_t1_ep_F)
        self.image_t1_ep_F_label.grid(row=0, column=0)
        self.textbox = customtkinter.CTkTextbox(self.tabview_1.tab("FashionMNIST"))
        self.textbox.grid(row=1, column=0, padx=(20, 0), pady=(20, 0), sticky="nsew")
        
        self.tabview_1.add("CATDOG")
        self.tabview_1.tab("CATDOG").grid_columnconfigure(0, weight=1)
        self.tabview_1.tab("CATDOG").grid_rowconfigure(0, weight=1)
        self.image_t1_ep_C = customtkinter.CTkImage(Image.open(os.path.join(current_path, "CATDOG_example_plot.png")), size=(1280, 150))
        self.image_t1_ep_C_label = customtkinter.CTkLabel(self.tabview_1.tab("CATDOG"), text ="", image=self.image_t1_ep_C)
        self.image_t1_ep_C_label.grid(row=0, column=0)
        self.textbox = customtkinter.CTkTextbox(self.tabview_1.tab("CATDOG"))
        self.textbox.grid(row=1, column=0, padx=(20, 0), pady=(20, 0), sticky="nsew")

        #Tabframe 2 - Training parameters
        self.tab_frame_2 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_2.grid_columnconfigure(0, weight=1)
        self.tab_frame_2.grid_rowconfigure(0, weight=1)
        self.tabview_2 = customtkinter.CTkTabview(self.tab_frame_2)
        self.tabview_2.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        self.tabview_2.add("Patience")
        self.tabview_2.tab("Patience").grid_columnconfigure(0, weight=1)
        self.tabview_2.tab("Patience").grid_rowconfigure(0, weight=1)

        self.tabview_2.add("Kernel")
        self.tabview_2.tab("Kernel").grid_columnconfigure(0, weight=1)
        self.tabview_2.tab("Kernel").grid_rowconfigure(0, weight=1)

        #Tabframe 3 - Small models
        self.tab_frame_3 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_3.grid_columnconfigure(0, weight=1)
        self.tab_frame_3.grid_rowconfigure(0, weight=1)
        self.tabview_3 = customtkinter.CTkTabview(self.tab_frame_3)
        self.tabview_3.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")
        
        self.tabview_3.add("Patience - 3")
        self.tabview_3.tab("Patience - 3").grid_columnconfigure(0, weight=1)
        self.tabview_3.tab("Patience - 3").grid_rowconfigure(0, weight=1)

        self.tabview_3.add("Patience - 5")
        self.tabview_3.tab("Patience - 5").grid_columnconfigure(0, weight=1)
        self.tabview_3.tab("Patience - 5").grid_rowconfigure(0, weight=1)

        self.tabview_3.add("Patience - 7")
        self.tabview_3.tab("Patience - 7").grid_columnconfigure(0, weight=1)
        self.tabview_3.tab("Patience - 7").grid_rowconfigure(0, weight=1)

        #Tabframe 4 - Medium models
        self.tab_frame_4 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_4.grid_columnconfigure(0, weight=1)
        self.tab_frame_4.grid_rowconfigure(0, weight=1)
        self.tabview_4 = customtkinter.CTkTabview(self.tab_frame_4)
        self.tabview_4.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")

        self.tabview_4.add("Patience - 3")
        self.tabview_4.tab("Patience - 3").grid_columnconfigure(0, weight=1)
        self.tabview_4.tab("Patience - 3").grid_rowconfigure(0, weight=1)

        self.tabview_4.add("Patience - 5")
        self.tabview_4.tab("Patience - 5").grid_columnconfigure(0, weight=1)
        self.tabview_4.tab("Patience - 5").grid_rowconfigure(0, weight=1)

        self.tabview_4.add("Patience - 7")
        self.tabview_4.tab("Patience - 7").grid_columnconfigure(0, weight=1)
        self.tabview_4.tab("Patience - 7").grid_rowconfigure(0, weight=1)

        #Tabframe 5 - Big models
        self.tab_frame_5 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_5.grid_columnconfigure(0, weight=1)
        self.tab_frame_5.grid_rowconfigure(0, weight=1)
        self.tabview_5 = customtkinter.CTkTabview(self.tab_frame_5)
        self.tabview_5.grid(row=0, column=0, padx=(0, 0), pady=(0, 0), sticky="nsew")

        self.tabview_5.add("Patience - 3")
        self.tabview_5.tab("Patience - 3").grid_columnconfigure(0, weight=1)
        self.tabview_5.tab("Patience - 3").grid_rowconfigure(0, weight=1)

        self.tabview_5.add("Patience - 5")
        self.tabview_5.tab("Patience - 5").grid_columnconfigure(0, weight=1)
        self.tabview_5.tab("Patience - 5").grid_rowconfigure(0, weight=1)

        self.tabview_5.add("Patience - 7")
        self.tabview_5.tab("Patience - 7").grid_columnconfigure(0, weight=1)
        self.tabview_5.tab("Patience - 7").grid_rowconfigure(0, weight=1)

        #Tabframe 6 - Test
        self.tab_frame_6 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_6.grid_columnconfigure(0, weight=1)
        self.tab_frame_6.grid_rowconfigure(0, weight=1)

        

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
 
if __name__ == "__main__":
    app = App()
    app.mainloop()