
from . import ResNet18, ResNet50, ResNet32, ResNet101, ResNet152



class ResNet18v2(ResNet18):
    """Resnet18v2
    """
    def __init__(self):
        """Initialize resnet18v2
        """
        super().__init__()

    
    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        super().set_filters_and_shortcuts()

        # resnet v2 uses preactivation
        self.preact = True


class ResNet32v2(ResNet32):
    """Resnet32v2
    """
    def __init__(self):
        """Initialize resnet32v2
        """
        super().__init__()

    
    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        super().set_filters_and_shortcuts()

        # resent v2 uses preactivation
        self.preact = True



class ResNet50v2(ResNet50):
    """Resnet50
    """
    def __init__(self):
        """Initialize resnet50
        """
        super().__init__()

    
    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        super().set_filters_and_shortcuts()

        # resnet v2 uses preactivation
        self.preact = True


class ResNet101v2(ResNet101):
    """Resnet101v2
    """
    def __init__(self):
        """Initialize resnet101v2
        """
        super().__init__()

    
    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        super().set_filters_and_shortcuts()

        # resnet v2 uses preactivation
        self.preact = True



class ResNet152v2(ResNet152):
    """Resnet152v2
    """
    def __init__(self):
        """Initialize resnet152v2
        """
        super().__init__()

    
    def set_filters_and_shortcuts(self):
        """Overides the parent method
        """
        super().set_filters_and_shortcuts()

        # preactivation
        self.preact = True



