# This is a sample Python script.
from Prover import Clause, Predicate, resolution, parse_cnf, parse_clause, parse_predicate, parse_clauses, to_cnf, \
    to_nnf
import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import ttk, Text
from tkinter import messagebox, scrolledtext

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    last_focused_clause = None


def prove():
    # Get the input type selected by the user
    input_type = input_var.get()
    parsed_clauses = set()

    try:
        if input_type == "CNF":
            # Get CNF input from the single text box
            cnf_input = fol_entry.get("1.0", tk.END).strip()
            parsed_clauses = parse_cnf(cnf_input)  # Parse CNF input directly

        elif input_type == "Clauses":
            # Gather all clauses from each clause entry text box
            for entry in clause_entries:
                clause_text = entry.get("1.0", tk.END).strip()
                if clause_text:  # Ensure clause is not empty
                    parsed_clause = parse_clause(clause_text)  # Parse each clause individually
                    parsed_clauses.add(parsed_clause)  # Add parsed clause to the list

        elif input_type == "FOL":
            # Get FOL input from the text box
            fol_input = fol_entry.get("1.0", tk.END).strip()
            # Convert the FOL to CNF
            # cnf_input = to_nnf(fol_input)
            cnf_input = to_cnf(fol_input)
            # print(cnf_input)
            # Parse the CNF after conversion
            parsed_clauses = parse_cnf(cnf_input)
            # print(parsed_clauses)

        # Call the resolution function and display the result
        result = resolution(parsed_clauses)
        result_label.config(text="Result: " + ("Provable" if result else "Not Provable"))

    except Exception as e:
        # Show an error message if parsing or proving fails
        messagebox.showerror("Error", str(e))


# Function to add a clause entry box
def add_clause_entry(parent):
    """Add a new clause input box."""
    # global last_focused_clause
    entry = tk.Text(parent, height=2, width=60, font=("Segoe UI", 12), bg="#eeeeee", fg="#333333", borderwidth=0,
                    highlightthickness=1, highlightbackground="#cccccc")
    add_placeholder_to_text(entry, "Enter your Clause formula here... \nEX: ¬man(x) ∨ mortal(x)")

    clause_entries.append(entry)
    entry.pack(pady=5)
    on_input_type_change()


# Function to remove the last clause entry box
def remove_clause_entry(parent):
    """Remove the last clause input box."""
    if clause_entries:
        last_entry = clause_entries.pop()
        last_entry.pack_forget()
        on_input_type_change()


def add_placeholder_to_text(text_widget, placeholder_text):
    text_widget.insert("1.0", placeholder_text)
    text_widget.config(fg="grey")

    def on_focus_in(event):
        # When the user clicks the input box, if the content is a placeholder, clear the text and set the color to black
        if text_widget.get("1.0", "end-1c").strip() == placeholder_text:
            text_widget.delete("1.0", "end")
            text_widget.config(fg="black")

        # Update the global last_focused_clause
        global last_focused_clause
        last_focused_clause = text_widget

    def on_focus_out(event):
        # When the input box loses focus and is empty, restore the placeholder text and color
        if not text_widget.get("1.0", "end-1c").strip():  # Check if input is empty
            text_widget.insert("1.0", placeholder_text)
            text_widget.config(fg="grey")

    # Binding the event
    text_widget.bind("<FocusIn>", on_focus_in)
    text_widget.bind("<FocusOut>", on_focus_out)


# Function to dynamically update input type
def on_input_type_change():

    # Clear the result when change input type
    result_label.config(text="Result: ")

    # Clear all widgets inside the input_frame
    for widget in input_frame.winfo_children():
        widget.pack_forget()

    # Rebuild the layout based on the selected mode
    if input_var.get() == "CNF" or input_var.get() == "FOL":
        fol_entry.pack(pady=10)  # Show CNF input
    else:
        for entry in clause_entries:
            entry.pack(pady=5)  # Show each clause entry
        add_clause_button.pack(pady=5)  # Show Add button
        remove_clause_button.pack(pady=5)  # Show Remove button

    # Update the symbol toolbar dynamically
    update_symbol_toolbar()

    # Ensure result label is always at the bottom
    result_label.pack(pady=10)


def open_fol_tutorial():
    tutorial_window = tk.Toplevel(root)
    tutorial_window.title("FOL Tutorial")
    tutorial_window.geometry("600x400")

    tutorial_label = tk.Label(tutorial_window, text="FOL Tutorial", font=("Arial", 18, "bold"))
    tutorial_label.pack(pady=10)

    tutorial_text = tk.Text(tutorial_window, wrap="word", font=("Segoe UI", 12))
    tutorial_text.insert(
        "1.0",
        """First Order Logic (FOL) Tutorial:

    FOL allows quantifiers and predicates to express logic more formally.

    Press the buttons to write a symbol.
    Symbols:
    - ∧ : AND
    - ∨ : OR
    - ¬ : NOT
    - -> : IMPLIES 
    - <->: IFF (If and only if)
    - ∀ : For All (Universal Quantifier) (Will be implemented in future releases)
    - ∃ : There Exists (Existential Quantifier) (Will be implemented in future releases)

    Constants: a, b, c, d, e

    Example:
    - (¬man(x) ∨ mortal(x)) ∧ man(a) → mortal(a)
    """)
    tutorial_text.config(state="disabled")
    tutorial_text.pack(expand=True, fill="both", padx=10, pady=10)


def open_cnf_tutorial():
    tutorial_window = tk.Toplevel(root)
    tutorial_window.title("CNF Tutorial")
    tutorial_window.geometry("600x400")

    tutorial_label = tk.Label(tutorial_window, text="CNF Tutorial", font=("Arial", 18, "bold"))
    tutorial_label.pack(pady=10)

    tutorial_text = tk.Text(tutorial_window, wrap="word", font=("Segoe UI", 12))
    tutorial_text.insert(
        "1.0",
        """Conjunctive Normal Form (CNF) Tutorial:

    CNF is a conjunction of one or more disjunctions of literals.

    Press the buttons to write a symbol.
    Symbols:
    - ∧ : AND
    - ∨ : OR
    - ¬ : NOT

    Constants: a, b, c, d, e

    Example CNF Formula:
    - (¬man(x) ∨ mortal(x)) ∧ man(a) ∧ ¬mortal(a)
    """)

    tutorial_text.config(state="disabled")
    tutorial_text.pack(expand=True, fill="both", padx=10, pady=10)


def open_clauses_tutorial():
    """Open the Clauses Tutorial"""
    tutorial_window = tk.Toplevel(root)
    tutorial_window.title("Clauses Tutorial")
    tutorial_window.geometry("600x400")

    tutorial_label = tk.Label(tutorial_window, text="Clauses Tutorial", font=("Arial", 14, "bold"))
    tutorial_label.pack(pady=10)

    tutorial_text = tk.Text(tutorial_window, wrap="word", font=("Segoe UI", 12))
    tutorial_text.insert(
        "1.0",
        """Clauses Tutorial:

    A clause is a disjunction of literals.

    Constants: a, b, c, d, e
    Press the buttons to write a symbol.

    Example:
    - Clause 1: ¬man(x) ∨ mortal(x)
    - Clause 2: man(a)
    - Clause 3: ¬mortal(a)

    When used together, these clauses can prove logical statements through resolution.
    """)
    tutorial_text.config(state="disabled")
    tutorial_text.pack(expand=True, fill="both", padx=10, pady=10)


def open_about_us():
    credits_window = tk.Toplevel(root)
    credits_window.title("About Us")
    credits_window.geometry("400x300")

    credits_label = tk.Label(credits_window, text="About Us", font=("Arial", 14, "bold"))
    credits_label.pack(pady=10)

    credits_text = tk.Label(
        credits_window,
        text=(
            "Our team members:\n"
            "Aswin Budi Rahardja - 301439005\n"
            "Bowei Pan - 301435285\n"
            "Linda Li - 301406637\n"
            "UCheng Hong - 301435964"
        ),
        font=("Segoe UI", 12),
        justify="left"
    )
    credits_text.pack(pady=20)


def insert_symbol(symbol):
    global last_focused_clause
    try:
        # Check if we're in Clauses mode and there are clause text boxes
        if input_var.get() == "Clauses":
            if clause_entries:
                # Use the last focused clause if available, otherwise default to the last added clause
                focused_widget = last_focused_clause if last_focused_clause in clause_entries else clause_entries[-1]
            else:
                messagebox.showerror("Warning", "Please select a valid text box.")
        else:
            # Default to the FOL/CNF text box
            focused_widget = fol_entry

        # Insert the symbol into the focused widget
        if isinstance(focused_widget, tk.Text):
            current_text = focused_widget.get("1.0", "end-1c").strip()
            if current_text.startswith("Enter your Clause formula here...") or current_text.startswith(
                    "Enter your FOL/CNF formula here..."):
                focused_widget.delete("1.0", "end")  # Remove placeholder text

            focused_widget.insert(tk.INSERT, symbol)  # Insert the symbol
            focused_widget.config(fg="black")
        else:
            messagebox.showwarning("Warning", "Please select a valid text box.")
    except Exception as e:
        messagebox.showerror("Error", f"Could not insert symbol: {str(e)}")


def create_symbol_buttons(symbols, parent_frame):
    button_frame = tk.Frame(parent_frame, bg="#f5f5f5")
    button_frame.pack(pady=5)

    for symbol in symbols:
        btn = ttk.Button(button_frame, text=symbol, width=3, command=lambda s=symbol: insert_symbol(s))
        btn.pack(side="left", padx=2, pady=2)

    return button_frame


def update_symbol_toolbar():
    for widget in toolbar_frame.winfo_children():
        widget.destroy()

    # Define symbols for each page
    symbols = []
    if input_var.get() == "FOL":
        symbols = ["∧", "∨", "¬", "->", "<->"]  # , "∀", "∃"]
    elif input_var.get() == "CNF":
        symbols = ["∧", "∨", "¬"]
    elif input_var.get() == "Clauses":
        symbols = ["∨", "¬"]

    # Add new buttons
    create_symbol_buttons(symbols, toolbar_frame)


# Set up the main application window
root = ThemedTk(theme="breeze")
root.title("FOL Prover")
root.minsize(600, 400)
root.configure(bg="#f5f5f5")
style = ttk.Style(root)
# style.theme_use("clam")
style.configure("TLabel", font=("Segoe UI", 12), foreground="#333333", background="#f5f5f5")
style.configure("TButton", font=("Segoe UI", 12), padding=10)
style.configure("TRadiobutton", font=("Segoe UI", 12), padding=5, background="#f5f5f5")

# Header Label
header_label = ttk.Label(root, text="First Order Logic Prover", font=("Arial", 18, "bold"))
header_label.pack(pady=10)

# Label for instructions
instruction_label = ttk.Label(root, text="Choose FOL or CNF or Clauses:", font=("Arial", 12))
instruction_label.pack(pady=5)

# Variable to store the user's input type selection
input_var = tk.StringVar(value="CNF")  # Default to CNF
input_var.trace("w", lambda *args: on_input_type_change())  # Trigger UI update on change

selector_frame = tk.Frame(root, bg="#f5f5f5")
selector_frame.pack(pady=5)

# Radio buttons for selecting input type
cnf_radio = ttk.Radiobutton(selector_frame, text="CNF", variable=input_var, value="CNF")
clauses_radio = ttk.Radiobutton(selector_frame, text="Clauses", variable=input_var, value="Clauses")
FOL_radio = ttk.Radiobutton(selector_frame, text="FOL", variable=input_var, value="FOL")
FOL_radio.grid(row=0, column=0, padx=10)
cnf_radio.grid(row=0, column=1, padx=10)
clauses_radio.grid(row=0, column=2, padx=10)

# Frame to hold input widgets and result label
input_frame = tk.Frame(root, bg="#f5f5f5")
input_frame.pack(pady=10)

# Text box for CNF input
fol_entry = tk.Text(input_frame, height=5, width=60, font=("Segoe UI", 12), bg="#eeeeee", fg="#333333", borderwidth=0,
                    highlightthickness=1, highlightbackground="#cccccc")
fol_entry.pack(pady=10)
add_placeholder_to_text(fol_entry,
                        "Enter your FOL/CNF formula here... \nEX: (¬man(x) ∨ mortal(x)) ∧ man(a) ∧ ¬mortal(a)")

# List to store multiple clause entries (used if "Clauses" is selected)
clause_entries = []

# Buttons to add or remove clause entry text boxes
add_clause_button = ttk.Button(input_frame, text="Add Clause", command=lambda: add_clause_entry(input_frame))
remove_clause_button = ttk.Button(input_frame, text="Remove Clause", command=lambda: remove_clause_entry(input_frame))

# Create a frame for the toolbar
toolbar_frame = tk.Frame(root, bg="#f5f5f5")
toolbar_frame.pack(pady=5)

# Initialize the toolbar
update_symbol_toolbar()

# Frame to hold input widgets and result label
result_frame = tk.Frame(root, bg="#f5f5f5")
result_frame.pack(pady=10)

# Result Label
result_label = tk.Label(result_frame, text="Result: ", font=("Arial", 14, "bold"), bg="#f5f5f5", fg="#333")
result_label.pack(pady=10)

# Prove button
prove_button = ttk.Button(root, text="Prove", command=prove)
prove_button.pack(pady=15)

# Main menu
menu = tk.Menu(root)

# Help menu
help_menu = tk.Menu(menu, tearoff=0)
help_menu.add_command(label="FOL Tutorial", command=open_fol_tutorial)
help_menu.add_command(label="CNF Tutorial", command=open_cnf_tutorial)
help_menu.add_command(label="Clauses Tutorial", command=open_clauses_tutorial)
menu.add_cascade(label="Help", menu=help_menu)

# About us
about_menu = tk.Menu(menu, tearoff=0)
about_menu.add_command(label="About Us", command=open_about_us)
menu.add_cascade(label="About Us", menu=about_menu)

# Exit Menu
exit_menu = tk.Menu(menu, tearoff=0)
exit_menu.add_command(label="Exit", command=root.quit)
menu.add_cascade(label="Exit", menu=exit_menu)

root.config(menu=menu)

# Run the application
root.mainloop()