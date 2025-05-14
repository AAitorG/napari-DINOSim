from qtpy.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QFrame,
    QHBoxLayout,
    QPushButton,
)


class CollapsibleSection(QWidget):
    """A collapsible section widget with a header and a content area.

    Parameters
    ----------
    title : str
        The title of the section.
    parent : QWidget, optional
        The parent widget.
    collapsed : bool, optional
        Whether the section is initially collapsed. Default is False.
    """

    def __init__(self, title, parent=None, collapsed=False):
        super().__init__(parent)

        # Create layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Create header
        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_layout.setContentsMargins(5, 5, 5, 5)

        # Toggle button
        self.toggle_button = QPushButton()
        self.toggle_button.setMaximumWidth(20)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(not collapsed)
        self.toggle_button.clicked.connect(self.toggle_content)

        # Section title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Add to header layout
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()

        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(20, 5, 5, 5)  # Indent content

        # Add line separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        # Add widgets to main layout
        self.main_layout.addWidget(self.header_widget)
        self.main_layout.addWidget(line)
        self.main_layout.addWidget(self.content)

        # Set initial state
        self.update_toggle_button()
        if collapsed:
            self.content.setVisible(False)

    def toggle_content(self):
        """Toggle the visibility of the content area."""
        self.content.setVisible(self.toggle_button.isChecked())
        self.update_toggle_button()

    def update_toggle_button(self):
        """Update the toggle button text based on collapsed state."""
        if self.toggle_button.isChecked():
            self.toggle_button.setText("▼")  # Down arrow for expanded
        else:
            self.toggle_button.setText("►")  # Right arrow for collapsed

    def add_widget(self, widget):
        """Add a widget to the content area.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        """
        self.content_layout.addWidget(widget)

    @property
    def native(self):
        """Return self to maintain compatibility with magicgui."""
        return self
