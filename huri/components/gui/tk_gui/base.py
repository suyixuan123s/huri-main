import tkinter as tk
from tkinter.font import Font
from huri.components.gui.tk_gui.widgets.checkbutton import Checkbutton
from huri.components.gui.tk_gui.widgets.listbox import ListBox
from huri.components.gui.tk_gui.widgets.scale import Scale
from huri.components.gui.tk_gui.widgets.groups import EntryGrid


class GuiFrame(tk.Toplevel):

    def __init__(self, root, topmost=False, close_hidden=False, resizable=False, hidden=False):
        tk.Toplevel.__init__(self, root)
        self._close_hidden = close_hidden
        self._topmost = topmost
        self._resizable = resizable
        if hidden:
            self.withdraw()
        self.registeration()

    def add_title(self, text, pos=(0, 0)):
        fontStyle = Font(family="Lucida Grande", size=13)
        label_widget = tk.Label(self, justify=tk.LEFT, text=text, font=fontStyle)
        row, column = pos
        label_widget.grid(row=row, column=column)
        return label_widget

    def add_text(self, text, pos=(0, 0)):
        label_widget = tk.Label(self, text=text)
        row, column = pos
        label_widget.grid(row=row, column=column)
        return label_widget

    def add_checkbutton(self, text, callback, is_checked=True, pos=(0, 0)):
        checkbutton_widget = Checkbutton(parent=self,
                                         text=text,
                                         callback=callback,
                                         is_checked=int(is_checked))
        row, column = pos
        checkbutton_widget.grid(row=row, column=column)
        return checkbutton_widget

    def add_listbox(self, options, has_clear_button=False, pos=(0, 0)):
        listbox_widget = ListBox(parent=self,
                                 options=options,
                                 has_clear_button=has_clear_button)
        row, column = pos
        listbox_widget.grid(row=row, column=column)
        return listbox_widget

    def add_grid_entry(self, num_of_columns, headers=None, pos=(0, 0)):
        entry_grid_widget = EntryGrid(parent=self, num_of_columns=num_of_columns, headers=headers)
        row, column = pos
        entry_grid_widget.grid(row=row, column=column, columnspan=2, sticky='ew')
        return entry_grid_widget

    def add_button(self, text: str, command: callable, pos=(0, 0)):
        button_widget = tk.Button(master=self, text=text, command=command)
        row, column = pos
        button_widget.grid(row=row, column=column)
        return button_widget

    def add_scale(self, text="scale",
                  val_range=(0, 255),
                  default_value=None,
                  command: callable = lambda: None,
                  pos=(0, 0)):
        scale_widge = Scale(parent=self,
                            text=text,
                            val_range=val_range,
                            default_value=default_value,
                            callback=command)
        row, column = pos
        scale_widge.grid(row=row, column=column)
        return scale_widge

    def set_title(self, title):
        self.title(title)

    def set_size(self, width, height):
        self.geometry(f"{int(width)}x{int(height)}")

    def registeration(self):
        # self.tk.pack_slaves()
        if self._close_hidden:
            self.protocol("WM_DELETE_WINDOW", self.on_closing)
        if self._topmost:
            self.attributes("-topmost", 1)
        if self._resizable:
            self.resizable(True, True)
        else:
            self.resizable(False, False)

    def on_closing(self):
        self.withdraw()

    def show(self):
        self.deiconify()


if __name__ == '__main__':
    root = tk.Tk()
    child = GuiFrame(root)
    child.set_size(width=400, height=200)
    child.set_title("Hello world")
    child.add_title("sdfdsf")
    child.add_checkbutton("Test", lambda v: print(f"{v}Test"), pos=(0, 1))
    child.add_button(text="test", command=lambda: None, pos=(1, 0))
    child.add_scale(command=lambda: None, val_range=[0, 3.0], pos=(2, 0))
    gird_entry = child.add_grid_entry(4, headers=["x", "y", "z", "r"], pos=(3, 0))
    gird_entry.add_row()
    gird_entry.add_row()
    gird_entry.add_row()
    print(gird_entry.read())
    root.mainloop()
