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
        self.main_frame.grid_columnconfigure(0, weight=1)

        button1 = customtkinter.CTkButton(self.main_frame, text="Neural Operators", command=self.tab_event, width=200)
        button1.grid(row=1, column=0, padx=30, pady=(15, 15))

        button2 = customtkinter.CTkButton(self.main_frame, text="The Data", command=self.button_test, width=200)
        button2.grid(row=2, column=0, padx=30, pady=(15, 15))

        button3 = customtkinter.CTkButton(self.main_frame, text="Comparison", command=self.button_test, width=200)
        button3.grid(row=3, column=0, padx=30, pady=(15, 15))

        button4 = customtkinter.CTkButton(self.main_frame, text="Test", command=self.button_test, width=200)
        button4.grid(row=4, column=0, padx=30, pady=(15, 15))

        self.login_button = customtkinter.CTkButton(self.main_frame, text="Login", command=self.button_test, width=200)
        self.login_button.grid(row=5, column=0, padx=30, pady=(15, 15))

        # create tab frame (Another frame for testing)
        self.tab_frame = customtkinter.CTkFrame(self,corner_radius=0)
        self.tab_frame.grid_columnconfigure(0, weight=1)
        
        self.tabview = customtkinter.CTkTabview(self.tab_frame, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Neural Networks and Neural Operators")
        self.tabview.add("The Data")
        self.tabview.add("Comparison")
        self.tabview.add("Test")
        self.back_button = customtkinter.CTkButton(self.tab_frame, text="Back", command=self.back_event, width=200)
        self.back_button.grid(row=1, column=0, padx=30, pady=(15, 15))
    
    def tab_event(self):
        self.main_frame.grid_forget()
        self.tab_frame.grid(row=0, column=0, sticky="ns")

    def back_event(self):
        self.tab_frame.grid_forget()
        self.main_frame.grid(row=0, column=0, sticky="ns")
    
    def button_test(self):
        print("CLICK")
 
if __name__ == "__main__":
    app = App()
    app.mainloop()