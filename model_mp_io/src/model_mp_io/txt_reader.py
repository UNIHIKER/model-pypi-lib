import os

class TxtReader:
    def __init__(self, file_path: str = None):
        """
        Initialize the TxtReader class.

        Args:
            file_path (str, optional): Path to the text file to read. If provided and exists,
                reads the file skipping the first line header.
        """
        self.file_path = file_path
        self.lines = []
        self.number_buffer = []  # buffer for single numbers
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                # skip the first line (header) when reading
                lines = f.readlines()[1:]
            self.lines = lines   # store data into `lines` after skipping header
        else:
            self.lines = []

    def read_line(self):
        """
        Read one line from the stored lines or number buffer.

        Returns:
            str: The next line from lines or joined numbers from buffer.

        Raises:
            ValueError: If no more lines to read.
        """
        if self.number_buffer:
            result = ','.join(self.number_buffer)
            self.number_buffer = []  # 清空缓冲区
            return result
        
        if self.lines:
            return self.lines.pop(0).strip()
        else:
            raise ValueError("No more lines to read")

    def add_line(self, lines: str):
        """
        Add a single number to the number buffer.

        Args:
            lines (str): A string representing a single number (no commas).

        Returns:
            list: The updated lines list (though primarily updates buffer).

        Raises:
            ValueError: If the input is not a single number.
        """
        # If it's a single number (no comma), add it to number_buffer
        if ',' not in lines and lines.strip().replace('.', '').isdigit():
            self.number_buffer.append(lines.strip())
        else:
            raise ValueError("Invalid line format. Only single numbers are allowed.")
        return self.lines
    