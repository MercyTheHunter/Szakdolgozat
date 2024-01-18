from tkinter import *
import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

app = customtkinter.CTk()
app.geometry("1280x720") #HD
app.title("gui.py")



button1 = customtkinter.CTkButton(master=root, text="Neural Operators")
button1.place(relx=0.5, rely=0.2, anchor=CENTER)

button2 = customtkinter.CTkButton(master=root, text="The Data")
button2.place(relx=0.5, rely=0.3, anchor=CENTER)

button3 = customtkinter.CTkButton(master=root, text="Comparison")
button3.place(relx=0.5, rely=0.4, anchor=CENTER)

button4 = customtkinter.CTkButton(master=root, text="Test")
button4.place(relx=0.5, rely=0.5, anchor=CENTER)


app.mainloop()