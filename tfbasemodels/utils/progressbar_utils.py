import progressbar

def obtain_widget():
    """Creates the widget to use
    """
    widgets = [
        ' [', progressbar.Timer(format= 'elapsed time: %(elapsed)s'), '] ',
        progressbar.SimpleProgress(), 
        ' (', progressbar.ETA(), ') ',
    ]
    return widgets


def obtain_progressbar(max_value):
    """Creates a progress bar
    """
    widgets = obtain_widget()
    bar = progressbar.ProgressBar(max_value=max_value,  widgets=widgets).start()
    return bar