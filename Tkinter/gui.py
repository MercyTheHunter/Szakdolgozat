from tkinter import *
import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

class MainApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{1280}x{720}") #HD
        self.title("gui.py")

        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Neural Networks and Neural Operators")
        self.tabview.add("The Data")
        self.tabview.add("Comparison")
        self.tabview.add("Test")
    
    def button_test(self):
        print("CLICK")

    def button_callback(self):
        app = MainApp()

class SubApp1(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{1280}x{720}") #HD
        self.title("nn.py")

        button = customtkinter.CTkButton(master=self, text="Back", command=self.button_callback)
        button.place(relx=0.8, rely=0.8, anchor=RIGHT)
    
    def button_callback(self):
        app = SubApp1()
 
if __name__ == "__main__":
    app = MainApp()
    app.mainloop()