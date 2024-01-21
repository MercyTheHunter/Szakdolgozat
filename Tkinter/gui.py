from tkinter import *
import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

class App(customtkinter.CTk):
    width = 1280
    height = 720

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)
        self.title("gui.py")

        # create main frame
        self.main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=0, padx=(275, 0), pady=(225, 0), sticky="nesw")
        #self.main_frame.grid_columnconfigure(0, weight=1)

        self.button1 = customtkinter.CTkButton(self.main_frame, text="Neural Operators", command=self.tab_event_1, width=200)
        self.button1.grid(row=1, column=0, padx=30, pady=(15, 15))

        self.button2 = customtkinter.CTkButton(self.main_frame, text="The Data", command=self.tab_event_2, width=200)
        self.button2.grid(row=2, column=0, padx=30, pady=(15, 15))

        self.button3 = customtkinter.CTkButton(self.main_frame, text="Comparison", command=self.tab_event_3, width=200)
        self.button3.grid(row=3, column=0, padx=30, pady=(15, 15))

        self.button4 = customtkinter.CTkButton(self.main_frame, text="Test", command=self.tab_event_4, width=200)
        self.button4.grid(row=4, column=0, padx=30, pady=(15, 15))

        #Tab frames
        self.tab_frame_1 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_1.grid_columnconfigure(0, weight=1)
        self.tabview_1 = customtkinter.CTkTabview(self.tab_frame_1, width=250)
        self.tabview_1.grid(row=0, column=1, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.tabview_1.add("Neural Network")
        self.tabview_1.add("Neural Operator")
        self.tabview_1.add("Fourier Neural Operator")

        self.tab_frame_2 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_2.grid_columnconfigure(0, weight=1)
        self.tabview_2 = customtkinter.CTkTabview(self.tab_frame_2, width=250)
        self.tabview_2.grid(row=0, column=1, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.tabview_2.add("About The Dataset")
        self.tabview_2.add("Analyzing The Data")
        self.tabview_2.add("Splitting The Data")

        self.tab_frame_3 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_3.grid_columnconfigure(0, weight=1)
        self.tabview_3 = customtkinter.CTkTabview(self.tab_frame_3, width=250)
        self.tabview_3.grid(row=0, column=1, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.tabview_3.add("A Regular Neural Network")
        self.tabview_3.add("A Neural Network With The FNO")
        self.tabview_3.add("Learning Rate")
        self.tabview_3.add("Accuracy")

        self.tab_frame_4 = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame_4.grid_columnconfigure(0, weight=1)
        self.tabview_4 = customtkinter.CTkTabview(self.tab_frame_4, width=250)
        self.tabview_4.grid(row=0, column=1, padx=(0, 0), pady=(0, 0), sticky="nsew")
        self.tabview_4.add("Placeholder")


    def tab_event_1(self):
        self.main_frame.grid(row=0, column=0, padx=(275, 0), pady=(225, 0), sticky="nesw")
        self.tab_frame_2.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_1.grid(row=0, column=1, padx=(100, 0), pady=(225, 0), sticky="nsew")
    def tab_event_2(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_2.grid(row=0, column=1, padx=(100, 0), pady=(225, 0), sticky="nsew")
    def tab_event_3(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_2.grid_forget()
        self.tab_frame_4.grid_forget()
        self.tab_frame_3.grid(row=0, column=1, padx=(100, 0), pady=(225, 0), sticky="nsew")
    def tab_event_4(self):
        self.tab_frame_1.grid_forget()
        self.tab_frame_2.grid_forget()
        self.tab_frame_3.grid_forget()
        self.tab_frame_4.grid(row=0, column=1, padx=(100, 0), pady=(225, 0), sticky="nsew")
 
if __name__ == "__main__":
    app = App()
    app.mainloop()