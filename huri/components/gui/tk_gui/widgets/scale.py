from huri.components.gui.tk_gui.widgets.widget import Widget
import tkinter as  tk


class Scale(Widget):
    def __init__(self, parent,
                 text="scale bar",
                 horizontal=True,
                 val_range=(0, 255),
                 default_value=None,
                 callback: callable = None, **options):
        self._parent = parent
        super(Scale, self).__init__(self._parent)
        if default_value is None:
            default_value = (val_range[1] - val_range[0]) / 2
        self._var = tk.DoubleVar(value=default_value)
        self._orient = tk.HORIZONTAL if horizontal else tk.VERTICAL
        self._scale = tk.Scale(self, orient=self._orient,
                               variable=self._var,
                               from_=val_range[0],
                               to=val_range[1],
                               relief=tk.RAISED,
                               length=400,
                               sliderlength=20,
                               label=text,
                               resolution=(val_range[1] - val_range[0]) / 200,
                               tickinterval=(val_range[1] - val_range[0]) / 5,
                               **options)

        self._scale.bind("<ButtonRelease-1>", callback)
        self._scale.grid(row=0, column=0)
