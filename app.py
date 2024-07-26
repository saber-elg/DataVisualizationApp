import tkinter as tk
from tkinter import *
from turtle import width
import pandas as pd
import sys
from io import StringIO
from sklearn.impute import KNNImputer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from tkinter import scrolledtext
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
from random import choice


BUTTON_FONT = ("Arial", 13, "bold")
LABEL_FONT = ("Arial", 20, "bold")
USER_FONT = ("Arial", 14, "bold")
INFO_FONT = ("Arial", 10, "bold")
SMALL_FONT = ("Arial", 12, "normal")
COLORS = ['green', 'red', 'purple', 'brown', 'blue']



class Visualization:
    def __init__(self, window):
        self.window = window
        window.title("Data Visualization APP")
        window.attributes("-fullscreen", True) 
        window.resizable(width=False, height=False)
        # Adjusting the grid row and column configurations
        window.grid_rowconfigure(1, weight=1) 
        window.grid_columnconfigure(0, weight=1)  
        window.configure(bg="#d2e3ee")

        self.df = pd.DataFrame()

        self.bar_x_label = tk.StringVar()
        self.bar_y_label = tk.StringVar()
        self.scatter_x_name = tk.StringVar()
        self.scatter_y_name = tk.StringVar()
        self.pie_value_name = tk.StringVar()
        self.pie_group_name = tk.StringVar()
        self.line_name = tk.StringVar()
        
        self.ruban_frame = tk.Frame(window, bg="honeydew" )

        exit_image = Image.open("exit.png")
        used_exit = ImageTk.PhotoImage(exit_image)
        self.exit_button =tk.Button(self.ruban_frame, image=used_exit, bg="honeydew", bd=0, command=close_window)
        exit_button.image = used_exit 

        home_image = Image.open("home.png")
        home_image = home_image.resize((40, 40))
        used_home = ImageTk.PhotoImage(home_image)
        self.home_button =tk.Button(self.ruban_frame, image=used_home, bg="honeydew", bd=0, command=home)
        home_button.image = used_home 
        
        self.button_home_loading = tk.Button(self.ruban_frame, text="Load",relief="raised", font=("Helvetica", 12, "bold"), command=start, width=14, height=1,bg="#96CCA8")
        self.button_overview = tk.Button(self.ruban_frame, text="Overview",relief="raised", font=("Arial", 12, "bold"), command=describe, width=14, height=1,bg="#f9c68e")
        self.button_clean = tk.Button(self.ruban_frame, text="Clean",relief="raised", font=("Arial", 12, "bold"),command=main_clean,width=14, height=1,bg="#C0E1D1")
        self.button_explore = tk.Button(self.ruban_frame, text="Explore",relief="raised", font=("Arial", 12, "bold"), command=explore, width=14, height=1,bg="#69A297")
        self.button_analyse = tk.Button(self.ruban_frame, text="Analyse", relief="raised",font=("Arial", 12, "bold"), width=14, height=1,bg="#50808E")
        
        self.ruban_frame.place(relx=0, rely=1, relwidth=1,height=45, anchor="sw")
        self.home_button.pack(side="left", padx=10, pady=10)
        self.exit_button.pack(side="left", padx=10, pady=10)
        
        # ================================ TOP FRAME ================================ #

        self.top_frame = Frame(self.window)
        self.top_frame.place(x=2, y=0, width=1363, height=40)

        self.build_chart = Label(self.top_frame, text="DATA LOAD", justify="center", font=("Helvetica", 16, "bold"), fg="#00B09B")
        self.build_chart.place(x=500, y=2)

        # ================================ DASHBOARD AREA ================================ #
        # top left canvas: -----------------------------------------------------------
        self.bar_heading = Label(self.window, text="Bar Chart", font=SMALL_FONT, bg="ivory")
        self.bar_heading.place(x=550, y=45, width=355, height=18)

        self.bar_info = Frame(self.window, bg="ivory")
        self.bar_info.place(x=550, y=65, width=355, height=65)

        self.x_label = Label(self.bar_info, text="XLabel", font=SMALL_FONT, bg="ivory", bd=1)
        self.x_label.grid(row=0, column=0, padx=10)

        self.y_label = Label(self.bar_info, text="YLabel", font=SMALL_FONT, bg="ivory", bd=1)
        self.y_label.grid(row=1, column=0, padx=10)

        self.x_box = ttk.Combobox(self.bar_info, font=SMALL_FONT, justify="center", state="readonly",
                                  textvariable=self.bar_x_label)
        self.x_box.grid(row=0, column=1)

        self.y_box = ttk.Combobox(self.bar_info, font=SMALL_FONT, justify="center", state="readonly",
                                  textvariable=self.bar_y_label)
        self.y_box.grid(row=1, column=1)

        self.bar_draw_button = Button(self.bar_info, text="draw", justify="center", font=INFO_FONT, relief=RIDGE, bd=2,
                                      bg="ivory", cursor="hand2", width=5, command=self.draw_bar_chart)
        self.bar_draw_button.grid(row=0, column=2, padx=10)

        self.bar_clear_button = Button(self.bar_info, text="clean", justify="center", font=INFO_FONT, relief=RIDGE,
                                       bg="ivory", cursor="hand2", bd=2, width=5, command=self.clear_bar)
        self.bar_clear_button.grid(row=1, column=2, padx=10)

        # bar diagram replacement:
        self.top_left = Frame(self.window, bg="ivory")
        self.top_left.place(x=550, y=135, width=355, height=220)
        self.canvas_1 = Canvas(self.top_left, width=355, height=235, bg="ivory", relief=RIDGE)
        self.canvas_1.pack()
        self.fig_1 = None
        self.output_1 = None

        # top right canvas: ----------------------------------------------------------
        self.scatter_heading = Label(self.window, text="Scatter Plot", font=SMALL_FONT, bg="ivory")
        self.scatter_heading.place(x=920, y=45, width=355, height=17)

        self.scatter_info = Frame(self.window, bg="ivory")
        self.scatter_info.place(x=920, y=65, width=355, height=65)

        self.scatter_x_label = Label(self.scatter_info, text="XLabel", font=SMALL_FONT, bg="ivory", bd=1)
        self.scatter_x_label.grid(row=0, column=0, padx=10)

        self.scatter_y_label = Label(self.scatter_info, text="YLabel", font=SMALL_FONT, bg="ivory", bd=1)
        self.scatter_y_label.grid(row=1, column=0, padx=10)

        self.scatter_x_box = ttk.Combobox(self.scatter_info, font=SMALL_FONT, justify="center", state="readonly",textvariable=self.scatter_x_name)
        self.scatter_x_box.grid(row=0, column=1)

        self.scatter_y_box = ttk.Combobox(self.scatter_info, font=SMALL_FONT, justify="center", state="readonly",textvariable=self.scatter_y_name)
        self.scatter_y_box.grid(row=1, column=1)

        self.scatter_draw_button = Button(self.scatter_info, text="draw", justify="center", font=INFO_FONT,relief=RIDGE, bd=2, bg="ivory", cursor="hand2", width=5,command=self.draw_scatter_chart)
        self.scatter_draw_button.grid(row=0, column=2, padx=10)

        self.scatter_clean_button = Button(self.scatter_info, text="clean", justify="center", font=INFO_FONT,relief=RIDGE, bg="ivory", cursor="hand2", bd=2, width=5,command=self.clear_scatter)
        self.scatter_clean_button.grid(row=1, column=2, padx=10)

        # diagram replacement:
        self.top_right = Frame(self.window, bg="ivory")
        self.top_right.place(x=920, y=135, width=355, height=220)
        self.canvas_2 = Canvas(self.top_right, width=355, height=235, bg="ivory", relief=RIDGE)
        self.canvas_2.pack()
        self.fig_2 = None
        self.output_2 = None

        # bottom left canvas: --------------------------------------------------------
        self.pie_heading = Label(self.window, text="Pie Chart", font=SMALL_FONT, bg="ivory")
        self.pie_heading.place(x=550, y=362, width=355, height=18)

        self.pie_info = Frame(self.window, bg="ivory")
        self.pie_info.place(x=550, y=382, width=355, height=65)

        self.pie_x_label = Label(self.pie_info, text="Values", font=SMALL_FONT, bg="ivory", bd=1)
        self.pie_x_label.grid(row=0, column=0, padx=10)

        self.pie_y_label = Label(self.pie_info, text="GroupBy", font=SMALL_FONT, bg="ivory", bd=1)
        self.pie_y_label.grid(row=1, column=0, padx=10)

        self.pie_value_box = ttk.Combobox(self.pie_info, font=SMALL_FONT, justify="center", state="readonly",
                                          textvariable=self.pie_value_name)
        self.pie_value_box.grid(row=0, column=1)

        self.pie_group_box = ttk.Combobox(self.pie_info, font=SMALL_FONT, justify="center", state="readonly",
                                          textvariable=self.pie_group_name)
        self.pie_group_box.grid(row=1, column=1)

        self.pie_draw_button = Button(self.pie_info, text="draw", justify="center", font=INFO_FONT, relief=RIDGE,
                                      bd=2, bg="ivory", cursor="hand2", width=5, command=self.draw_pie_chart)
        self.pie_draw_button.grid(row=0, column=2, padx=10)

        self.pie_clear_button = Button(self.pie_info, text="clean", justify="center", font=INFO_FONT, relief=RIDGE,
                                       bg="ivory", cursor="hand2", bd=2, width=5, command=self.clear_pie)
        self.pie_clear_button.grid(row=1, column=2, padx=10)

        self.bottom_left = Frame(self.window, bg="ivory")
        self.bottom_left.place(x=550, y=452, width=355, height=220)
        self.canvas_3 = Canvas(self.bottom_left, width=355, height=235, bg="ivory", relief=RIDGE)
        self.canvas_3.pack()
        self.fig_3 = None
        self.output_3 = None

        # bottom right canvas: ------------------------------------------------------
        self.line_heading = Label(self.window, text="Line Chart", font=SMALL_FONT, bg="ivory")
        self.line_heading.place(x=920, y=362, width=355, height=18)

        self.line_info = Frame(self.window, bg="ivory")
        self.line_info.place(x=920, y=382, width=355, height=65)

        self.line_box = ttk.Combobox(self.line_info, font=SMALL_FONT, justify="center", state="readonly",
                                     textvariable=self.line_name)
        self.line_box.grid(row=0, column=1)

        self.line_draw_button = Button(self.line_info, text="draw", justify="center", font=INFO_FONT, relief=RIDGE,
                                       bd=2, bg="ivory", cursor="hand2", command=self.draw_line_chart)
        self.line_draw_button.grid(row=0, column=0, padx=10, pady=20)

        self.line_clear_button = Button(self.line_info, text="clean", justify="center", font=INFO_FONT, relief=RIDGE,
                                        bg="ivory", cursor="hand2", bd=2, command=self.clear_line)
        self.line_clear_button.grid(row=0, column=2, padx=10, pady=20)

        self.bottom_right = Frame(self.window, bg="ivory")
        self.bottom_right.place(x=920, y=452, width=355, height=220)
        self.canvas_4 = Canvas(self.bottom_right, width=355, height=240, bg="ivory", relief=RIDGE)
        self.canvas_4.pack()
        self.fig_4 = None
        self.output_4 = None
        

        # =================================== LEFT FRAME ================================ #
        self.left_frame = Frame(self.window, bg="white smoke", relief=RIDGE, bd=1)
        self.left_frame.place(x=2, y=45, width=550, height=630)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", backgroung="silver", foreground="black", rowheight=25, fieldbackground="silver")
        style.map("Treeview", background=[("selected", "medium sea green")])
        style.configure("Treeview.Heading", background="light steel blue", font=("Arial", 10, "bold"))

        self.my_table = ttk.Treeview(self.left_frame)

        scroll_x_label = ttk.Scrollbar(self.left_frame, orient=HORIZONTAL, command=self.my_table.xview)
        scroll_y_label = ttk.Scrollbar(self.left_frame, orient=VERTICAL, command=self.my_table.yview)
        scroll_x_label.pack(side=BOTTOM, fill=X)
        scroll_y_label.pack(side=RIGHT, fill=Y)
        
        self.file_open()

    # ================================= FUNCTIONALITY =============================== #
    
            
    def file_open(self):
        self.df=df

        # clean existing table:
        self.clear_table_data()
        # from csv into dataframe:
        self.my_table["column"] = list(self.df.columns)
        self.my_table["show"] = "headings"
        for column in self.my_table["column"]:
            self.my_table.heading(column, text=column)
        # resize columns:
        for column_name in self.my_table["column"]:
            self.my_table.column(column_name, width=60)
        # fill rows with data:
        df_rows_old = self.df.to_numpy()
        df_rows_refreshed = [list(item) for item in df_rows_old]
        for row in df_rows_refreshed:
            self.my_table.insert("", "end", values=row)
        self.my_table.place(x=5, y=5, width=530, height=630)
        try:
            self.fill_scatter_box()
        except TclError:
            pass

        try:
            self.fill_bar_box()
        except TclError:
            pass

        try:
            self.fill_pie_box()
        except TclError:
            pass

        try:
            self.fill_line_box()
        except TclError:
            pass

    def clear_table_data(self):
        self.my_table.delete(*self.my_table.get_children())

    # ================================ FILL COMBOBOX METHODS ============================= #
    def fill_bar_box(self):
        columns = [item for item in self.df]
        x_labels = []
        y_labels = []
        for column in columns:
            if self.df[column].dtype == 'object':
                x_labels.append(column)
            elif self.df[column].dtype == 'int64' or self.df[column].dtype == 'float64':
                y_labels.append(column)
        self.x_box["values"] = tuple(x_labels)
        self.x_box.current(0)
        self.y_box["values"] = tuple(y_labels)
        self.y_box.current(0)

    def fill_scatter_box(self):
        columns = [item for item in self.df]
        x_labels = []
        y_labels = []
        for column in columns:
            if self.df[column].dtype == 'int64' or self.df[column].dtype == 'float64':
                x_labels.append(column)
                y_labels.append(column)
        self.scatter_x_box["values"] = tuple(x_labels)
        self.scatter_x_box.current(0)
        self.scatter_y_box["values"] = tuple(y_labels)
        self.scatter_y_box.current(0)

    def fill_pie_box(self):
        columns = [item for item in self.df]
        x_labels = []
        y_labels = []
        for column in columns:
            if self.df[column].dtype == 'object':
                x_labels.append(column)
            elif self.df[column].dtype == 'int64' or self.df[column].dtype == 'float64':
                y_labels.append(column)
        self.pie_group_box["values"] = tuple(x_labels)
        self.pie_group_box.current(0)
        self.pie_value_box["values"] = tuple(y_labels)
        self.pie_value_box.current(0)

    def fill_line_box(self):
        columns = [item for item in self.df]
        x_labels = []
        for column in columns:
            if self.df[column].dtype == 'int64' or self.df[column].dtype == 'float64':
                x_labels.append(column)
        self.line_box["values"] = tuple(x_labels)
        self.line_box.current(0)

    # =================================== DRAW CHARTS ========================== #
    def draw_bar_chart(self):
        self.fig_1 = Figure(figsize=(4, 2), dpi=100)
        axes = self.fig_1.add_subplot(111)
        axes.bar(self.df[f"{self.bar_x_label.get()}"], self.df[f"{self.bar_y_label.get()}"], color=choice(COLORS))
        self.output_1 = FigureCanvasTkAgg(self.fig_1, master=self.canvas_1)
        self.output_1.draw()
        self.output_1.get_tk_widget().pack()

    def clear_bar(self):
        if self.output_1:
            for child in self.canvas_1.winfo_children():
                child.destroy()
        self.output_1 = None

    def draw_scatter_chart(self):
        self.fig_2 = Figure(figsize=(4, 2), dpi=100)
        axes = self.fig_2.add_subplot(111)
        axes.scatter(self.df[f"{self.scatter_x_name.get()}"], self.df[f"{self.scatter_y_name.get()}"], c=choice(COLORS))
        self.output_2 = FigureCanvasTkAgg(self.fig_2, master=self.canvas_2)
        self.output_2.draw()
        self.output_2.get_tk_widget().pack()

    def clear_scatter(self):
        if self.output_2:
            for child in self.canvas_2.winfo_children():
                child.destroy()
        self.output_2 = None

    def draw_pie_chart(self):
        # prepare values:
        display = self.df.groupby([f"{self.pie_group_name.get()}"]).sum(numeric_only=True)
        display = display[f"{self.pie_value_name.get()}"].to_numpy()
        my_labels = list(self.df[f"{self.pie_group_name.get()}"].unique())
        # visualize:
        self.fig_3 = Figure(figsize=(4, 2), dpi=100)
        axes = self.fig_3.add_subplot(111)
        axes.pie(display, labels=my_labels, shadow=True)
        self.output_3 = FigureCanvasTkAgg(self.fig_3, master=self.canvas_3)
        self.output_3.draw()
        self.output_3.get_tk_widget().pack()

    def clear_pie(self):
        if self.output_3:
            for child in self.canvas_3.winfo_children():
                child.destroy()
        self.output_3 = None

    def draw_line_chart(self):
        self.fig_4 = Figure(figsize=(4, 2), dpi=100)
        axes = self.fig_4.add_subplot(111)
        axes.plot(self.df[f"{self.line_name.get()}"], c=choice(COLORS))
        self.output_4 = FigureCanvasTkAgg(self.fig_4, master=self.canvas_4)
        self.output_4.draw()
        self.output_4.get_tk_widget().pack()

    def clear_line(self):
        if self.output_4:
            for child in self.canvas_4.winfo_children():
                child.destroy()
        self.output_4 = None

    def close_window(self):
        confirm = messagebox.askyesno(title="Data Visualization", message="Do You Want To Close Program?")
        if confirm > 0:
            self.window.destroy()
            return
        else:
            pass


    

# =============================== DATA LOADING PAGE =========================== #
def load():
    loading_text = "Loading..."
    widget_load.delete('1.0', tk.END)
    widget_load.insert('1.0', loading_text)
    fen.after(000, load_data)

def load_data():
    global df
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel Files", "*.xlsx;*.xls;*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if file_path:
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t')

        if df is not None and not df.empty:
            # Create a table
            table_data = format_columns(df.columns) + "\n"
            for index, row in df.iterrows():
                table_data += format_columns(row.values) + "\n"

            # Clear current content and insert the table
            widget_load.insert('1.0', table_data)
            button_overview.grid(row=0, column=1, padx=5, pady=5)
            # Show the number of rows and columns
            rows_cols_text = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
            rows_cols_label.config(text=rows_cols_text)
def format_columns(data):
    max_width = 20  
    formatted_data = ""
    for item in data:
        formatted_data += str(item).ljust(max_width)[:max_width]
    return formatted_data

# =============================== DATA DESCRIPTION PAGE  ======================= #
def describe():
    for widget in fen.winfo_children():
        widget.grid_remove()
    title_load.destroy()
    global title_description
    title_description = tk.Label(fen, text="DATA DESCRIPTION", justify="center", font=("Helvetica", 16, "bold"), fg="#00B09B")
    title_description.grid(row=0, column=0, columnspan=2, pady=10,sticky="ew")
    ruban_frame.grid(column=0,columnspan=2, sticky="sew")
    create_three_text_frames(fen, df)
    
    home_button.grid(row=0, column=0, padx=5, pady=5)
    button_home_loading.grid(row=0,column=1 ,padx=5, pady=5)
    button_overview.grid(row=0, column=2, padx=5, pady=5)
    button_clean.grid(row=0,column=3, padx=5, pady=5)
    exit_button.grid(row=0, column=6, padx=5, pady=5)
    

def create_text_frame(parent_frame, title, text_content, row, column,rowspan,height,width):
    frame = tk.Frame(parent_frame, bg="white", bd=1, relief="solid")
    title_label = tk.Label(frame, text=title, font=("Helvetica", 12, "bold"),fg="#5B61A1")
    title_label.grid(row=0, column=0, columnspan=2, sticky="ew", padx=15, pady=5)
    widget_load = tk.Text(frame, wrap="word", height=height,width=width)
    widget_load.insert('1.0', text_content)
    widget_load.config(state="disabled")

    # scrollbars for frames..................................................
    
    y_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=widget_load.yview)
    widget_load.config(yscrollcommand=y_scrollbar.set)
    y_scrollbar.grid(row=1, column=1, sticky="ns")
    x_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=widget_load.xview)
    widget_load.config(xscrollcommand=x_scrollbar.set)
    x_scrollbar.grid(row=2, column=0, sticky="ew")
    widget_load.grid(row=1, column=0, sticky="nsew")
    frame.grid(row=row, column=column, padx=10, pady=10, sticky="nsew",rowspan=rowspan)

def create_three_text_frames(parent_frame, df):
    description_content = df.describe().to_string()
    null_counts_content = df.isnull().sum().to_string()
    sys.stdout = StringIO()
    df.info()
    info_content = sys.stdout.getvalue()
    sys.stdout = sys.__stdout__

    create_text_frame(parent_frame, "DataFrame Description", description_content, 1, 0 ,rowspan=2,height=33,width=99)
    create_text_frame(parent_frame, "Null Counts", null_counts_content, 2, 1 , rowspan=1,height=14,width=50)
    create_text_frame(parent_frame, "DataFrame Info", info_content, 1, 1,rowspan=1,height=14,width=50)
   


# ============================== CLEANING DATA PAGE ============================ #
def clear_frame_clean():
    global frame_clean
    for widget in frame_clean.winfo_children():
        widget.destroy()

def create_checkboxes():
    global checkbox_frame
    global column_vars
    clear_frame_clean() 
    frame_clean.grid_rowconfigure(0, weight=1)  
    frame_clean.grid_columnconfigure(0, weight=1) 
    checkbox_frame = tk.Frame(frame_clean)
    checkbox_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
    # Checkboxes Creation
    for i, col_name in enumerate(df.columns):
        checkbox = tk.Checkbutton(checkbox_frame, text=col_name, variable=column_vars[i], activeforeground="#6190E8",)
        checkbox.grid(row=i+1, column=0,sticky="w",padx=200)  # Ajustement de l'index de ligne si nécessaire
    checkbox_frame.grid(row=0,  sticky="nsew")
    title_checkbox = tk.Label(checkbox_frame, text="Choose columns to drop:" ,font=("Arial", 24, "bold"),fg="#5B61A1")
    title_checkbox.grid(row=0,pady=20) 
    # Create the "Drop Selected Columns" button
    drop_button = tk.Button(checkbox_frame,width=40 ,height=4 ,text="Drop Selected Columns", command=drop_columns)
    drop_button.grid( padx=20, pady=20)

def drop_columns():
    global df
    global column_vars   
    selected_columns = [col for var, col in zip(column_vars, df.columns) if var.get() == 1]
    df = df.drop(selected_columns, axis=1)
    checkbox_frame.destroy()
    # Reset all checkboxes to unchecked state
    for var in column_vars:
        var.set(0)
    create_checkboxes()  # Update checkboxes after dropping columns

def drop_duplicates():
    global df
    df.drop_duplicates(inplace=True)
    messagebox.showinfo("Success", "Duplicated rows have been dropped successfully.")

def drop_rows():
    global frame_clean
    clear_frame_clean() 
    # Create message label
    message_label = tk.Label(frame_clean, text="Are you sure you want to drop duplicated rows?", font=("Arial", 24, "bold"), fg="#5B61A1")
    message_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    validate_button = tk.Button(frame_clean, text="Validate",width=10 , height=3 ,command=drop_duplicates)
    validate_button.grid(row=1, column=0, padx=20, pady=20)

def manage_nulls():
    global frame_clean
    # Create buttons on frame_clean
    button4 = tk.Button(frame_clean, text="Drop", width=15, height=1,command=drop_null_values)
    button5 = tk.Button(frame_clean, text="Replace", width=15, height=1,command=show_replace_options())
    button4.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    button5.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# null values Management ......................................................

def drop_null_values():
    global df
    df.dropna(inplace=True)
    messagebox.showinfo("Success", "Null values have been dropped successfully.")

def replace_with_mean():
    global df
    messagebox.showinfo("Success", "Null values have been replaced with max successfully.")
    df.fillna(df.mean(), inplace=True)
        
def replace_with_max():
    global df
    messagebox.showinfo("Success", "Null values have been replaced with max successfully.")
    df.fillna(df.max(), inplace=True)
    
def replace_with_kmeans():
    global df
    imputer = KNNImputer(n_neighbors=2)  
    messagebox.showinfo("Success", "Null values have been replaced with the methode KNN successfully.")
    df = imputer.fit_transform(df)

def replace_with_LOC():
    global df
    messagebox.showinfo("Success", "Null values have been replaced with the Last Observation Carried Forward successfully.")
    df = df.ffill()

def show_replace_options():
    global frame_clean
    clear_frame_clean()
    
    message_label = tk.Label(frame_clean, text="Choose what to replace with:",font=("Arial", 20, "bold"),fg="#5B61A1")
    message_label.grid(row=0, column=0, padx=60, pady=30, sticky="ew")
    # Create buttons for replacementoptions
    options = [("Mean", replace_with_mean), ("Max", replace_with_max), ("KNN", replace_with_kmeans), ("LOC", replace_with_LOC)] 
    for i, (option_text, command) in enumerate(options):
        if i == 0:  # Option 1
            button = tk.Button(frame_clean, text=option_text, width=15, height=2, command=lambda: (replace_with_mean()))
        if i == 1:
            button = tk.Button(frame_clean, text=option_text, width=15, height=2, command=lambda: (replace_with_max()))
        if i == 2:
            button = tk.Button(frame_clean, text=option_text, width=15, height=2, command=lambda: (replace_with_kmeans()))
        if i == 3:
            button = tk.Button(frame_clean, text=option_text, width=15, height=2, command=lambda: (replace_with_LOC()))
        button.grid(row=i+1, column=0, padx=70, pady=10,sticky="ew")
        
def main_clean():
    global frame_clean,column_vars,title_clean
    for widget in fen.winfo_children():
        if(isinstance(widget,tk.Toplevel)):
            widget.destroy
            continue
        widget.grid_remove()
        fen.update()
    ruban_frame.grid(row=4,column=0,columnspan=2, sticky="sew")
    title_description.destroy()
    title_clean = tk.Label(fen, text="DATA CLEANING", justify="center", font=("Helvetica", 16, "bold"), fg="#00B09B")
    title_clean.grid(row=0, columnspan=3, pady=10,sticky="ew")
    df.infer_objects()
    column_vars = [tk.IntVar() for _ in df.columns] 

    #buttons and frame creation
    button_frame = tk.Frame(fen, bg="white", bd=1, relief="solid")
    button1 = tk.Button(button_frame, text="Drop column", width=28, height=3, command=create_checkboxes)
    button2 = tk.Button(button_frame, text="Manage Null values", width=28, height=3, command=manage_nulls)
    button3 = tk.Button(button_frame, text="Manage duplicated rows", width=28, height=3, command=drop_rows)
    frame_clean = tk.Frame(fen,bg="white", bd=1, relief="solid",height=550, width=800)
    
    button_frame.grid(row=2, column=0, padx=5, pady=10, columnspan=3)
    button1.grid(row=2, column=0, padx=15, pady=5)  
    button2.grid(row=2, column=1, padx=15, pady=5)  
    button3.grid(row=2, column=2, padx=15, pady=5)
    
    frame_clean.propagate(False)
    frame_clean.grid_propagate(False)
    frame_clean.grid(row=1,column=0,columnspan=3)

    home_button.grid(row=0, column=0, padx=5, pady=5)
    button_home_loading.grid(row=0,column=1,padx=5,pady=5)
    button_overview.grid(row=0, column=2, padx=5, pady=5)
    button_clean.grid(row=0,column=3, padx=5, pady=5)
    button_explore.grid(row=0,column=4,padx=5,pady=5)
    exit_button.grid(row=0, column=5, padx=5, pady=5)


# =============================== DATA EXPLORATION PAGE ====================== #
def explore():
    global df
    global frame_clean, column_vars
    global title_explore

    for widget in fen.winfo_children():
        if(isinstance(widget,tk.Toplevel)):
            widget.destroy
            continue
        widget.grid_remove()
    ruban_frame.grid(column=0,columnspan=2, sticky="sew")
    title_clean.destroy()
    title_explore = tk.Label(fen, text="DATA EXPLORATION", justify="center", font=("Helvetica", 16, "bold"), fg="#00B09B")
    title_explore.grid(row=0, columnspan=2, pady=10,sticky="ew")
    
    df = pd.DataFrame(df)
    df_numeric = df.select_dtypes(include=[np.number])  # Select only numeric columns

    corr_matrix = df_numeric.corr()

    frame_expl = ttk.Frame(fen)
    frame_expl.grid(row=1, column=0, padx=10, pady=10)

    # Creation of figures and axes for the heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(4, 3))
    ax_corr.set_title("Heatmap", fontdict={'fontsize': 10, 'fontweight': 'bold'})
    sns.heatmap(corr_matrix, ax=ax_corr, annot=True, cmap="YlGnBu")
    heatmap_path = "Heatmap.png"
    plt.savefig(heatmap_path, format='png')
    plt.close()
    # Save the pairplot as an image
    fig_pair, ax_pair = plt.subplots(figsize=(4, 3))
    ax_pair.set_title("Pairplot", fontdict={'fontsize': 16, 'fontweight': 'bold'})
    sns.pairplot(df_numeric)
    pairplot_path = "pairplot.png"
    plt.savefig(pairplot_path, format='png')
    plt.close()

    # Display the heatmap on the first canvas
    image_corr = Image.open(heatmap_path)
    image_corr = image_corr.resize((650, 500))
    photo_corr = ImageTk.PhotoImage(image_corr)
    canvas_corr = Canvas(frame_expl, width=image_corr.width, height=image_corr.height)
    canvas_corr.create_image(0, 0, anchor=tk.NW, image=photo_corr)
    canvas_corr.grid(row=0, column=0, padx=10, pady=10)

    canvas_corr.photo = photo_corr

    # Display the pairplot image on the second canvas
    image = Image.open(pairplot_path)
    image = image.resize((650, 500))
    photo = ImageTk.PhotoImage(image)
    canvas_pair = Canvas(frame_expl, width=image.width, height=image.height)
    canvas_pair.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas_pair.grid(row=0, column=1, padx=10, pady=10)

    # Keep a reference to the photo to prevent it from being garbage collected
    canvas_pair.photo = photo
    home_button.grid(row=0, column=0, padx=5, pady=5)
    button_home_loading.grid(row=0,column=1,padx=5,pady=5)
    button_overview.grid(row=0, column=2, padx=5, pady=5)
    button_clean.grid(row=0,column=3, padx=5, pady=5)
    button_explore.grid(row=0,column=4,padx=5,pady=5)
    button_analyse.grid(row=0,column=5, padx=5, pady=5)
    exit_button.grid(row=0, column=6, padx=5, pady=5)


   
# ============================= DATA VISUALISATION PAGE ==================== #
def dashboard():
    top=tk.Toplevel()
    Visualization(top)
    
# exit command ........................................................
def close_window():
        confirm = messagebox.askyesno(title="Data Visualization APP", message="Do You Want To Close Program?")
        if confirm > 0:
            fen.destroy()
            return
        else:
            pass
# start command ........................................................
def start():
    global title_load
    for widget in fen.winfo_children():
        widget.grid_remove()
    set_win()
    fen.update()
    title_load = tk.Label(fen, text="DATA LOAD", justify="center", font=("Helvetica", 16, "bold"), fg="#00B09B")
    title_load.grid(row=0, columnspan=2,sticky="ew", pady=3)
    widget_load.grid(row=4, column=0)
    home_button.grid(row=0, column=0, padx=5, pady=5)
    rows_cols_label.grid(row=2, column=0, columnspan=2, sticky="e")
    scrollbar_horiz.grid(row=3, column=0, columnspan=2, sticky="ew")
    button_load.grid(row=2, column=0)
    scrollbar_vert.grid(row=3, column=1, rowspan=2, sticky="ns")
    widget_load.delete('1.0', tk.END)
    ruban_frame.grid(column=0,row=9,columnspan=2, sticky="sew")
    exit_button.grid(row=0, column=6, padx=5, pady=5)


def home(): 
    for widget in fen.winfo_children():
        if(isinstance(widget,tk.Toplevel)):
            widget.destroy
            continue
        widget.grid_remove()

    canvas = tk.Canvas(fen, width=window_width, height=window_height)
    canvas.grid(row=0, column=0)  
    image = Image.open("bg.png")
    image = image.resize((2600,1435 )) 
    global background_photo  
    background_photo = ImageTk.PhotoImage(image)

    canvas.create_image(0, 0, anchor=tk.CENTER, image=background_photo)
    
    canvas.create_text(window_width/2, 240, text="Welcome to", font=("Georgia", 15), fill="white")
    canvas.create_text(window_width/2, 300, text="DATA VISUALIZATION APP", font=("Palatino Linotype", 30, "bold"), fill="white")
    canvas.create_text(window_width/2, 420, text="About :  It's an application which offers exploring, cleaning, and analyzing your data through multiple graphs.\n\n\\t\t\tDeveloped by  : Soukaina El Hadifi & Mohamed Saber El Guelta", font=("Helvetica", 12, "italic"), fill="white")
    # Create the 'Start' button on canvas
    button_start = tk.Button(canvas, text="START", command=start, height=1, width=8, bg="pink", font=("Helvetica", 14, "bold"))
    canvas.create_window(window_width/2, 540, anchor=tk.CENTER, window=button_start)
    

      
fen = tk.Tk()
def set_win():
    fen.title("Data Visualization APP")
    fen.attributes("-fullscreen", True) 
    fen.resizable(width=False, height=False)
  
    fen.grid_rowconfigure(1, weight=1) 
    fen.grid_columnconfigure(0, weight=1)  
    fen.configure(bg="#d2e3ee")

# the main window ..........................................................
set_win()
fen.update()
window_width = fen.winfo_width()
window_height = fen.winfo_height()
home()

ruban_frame = tk.Frame(fen, bg="honeydew", height=100)

scrollbar_horiz = tk.Scrollbar(fen, orient="horizontal")
scrollbar_vert = tk.Scrollbar(fen, orient="vertical")
# definition of  buttons ...................................................
button_load = tk.Button(fen, text="Load", command=load, width=30, height=3)
widget_load = tk.Text(fen, height=35, width=window_width, wrap="none", xscrollcommand=scrollbar_horiz.set, yscrollcommand=scrollbar_vert.set)
button_start=tk.Button(ruban_frame, text="Load",relief="raised", font=("Helvetica", 12, "bold"), command=start, width=14, height=1,bg="#96CCA8")
button_home_loading = tk.Button(ruban_frame, text="Load",relief="raised", font=("Helvetica", 12, "bold"), command=start, width=14, height=1,bg="#96CCA8")
button_overview = tk.Button(ruban_frame, text="Overview",relief="raised", font=("Arial", 12, "bold"), command=describe, width=14, height=1,bg="#f9c68e")
button_clean = tk.Button(ruban_frame, text="Clean",relief="raised", font=("Arial", 12, "bold"),command=main_clean,width=14, height=1,bg="#C0E1D1")
button_explore = tk.Button(ruban_frame, text="Explore",relief="raised", font=("Arial", 12, "bold"), command=explore, width=14, height=1,bg="#69A297")
button_analyse = tk.Button(ruban_frame, text="Analyse", relief="raised",font=("Arial", 12, "bold"),command=dashboard, width=14, height=1,bg="#50808E")
# .........................................................................
rows_cols_label = tk.Label(fen, text="", width=20,bg="#d2e3ee")
frame_heatmap = ttk.Frame(fen)
frame_correlation = ttk.Frame(fen)

text_correlation = scrolledtext.ScrolledText(frame_correlation, width=40, height=20)


# Icones ..................................................................
  
exit_image = Image.open("exit.png")
used_exit = ImageTk.PhotoImage(exit_image)
exit_button =tk.Button(ruban_frame, image=used_exit, bg="honeydew", bd=0, command=close_window)
exit_button.image = used_exit 

home_image = Image.open("home.png")
home_image = home_image.resize((40, 40))
used_home = ImageTk.PhotoImage(home_image)
home_button =tk.Button(ruban_frame, image=used_home, bg="honeydew", bd=0, command=home)
home_button.image = used_home 

fen.mainloop()