import chardet
from dataclasses import dataclass
from pathlib import Path
import re

import yaml


@dataclass
class FilesLoader:
    """Reads paths, file names, opens and read the content of files to automatically detect the encoding of the data. 
    Indicates whether the encoding found across all files within a directory is consistent. 
    Stores all of the info above in class instance attributes.
    """

    config: dict
    extension: str

    def __post_init__(self):
        self.dir_path = Path(config["data_path"])
        self.file_names = list(self.dir_path.glob(self.extension))
        self.files_and_encoding = dict.fromkeys(self.file_names, 0)

        for file in self.file_names:
            with open(file, "rb+") as f:
                data = f.read()
                encoding_detected = chardet.detect(data)
                self.files_and_encoding[file] = encoding_detected

        self.encoding_isConsistent = (
            len(set([i["encoding"] for i in self.files_and_encoding.values()])) <= 1
        )


class EncodedTextFilesParser:
    """
    Uses the class attributes from FilesLoader to: \n
    --open the files, \n
    --decode their content with the automatically detected encoding format, \n
    --clean up unwanted characters (e.g. unicode)\n

    Returns: a list of documents in raw string format.
    """

    def __init__(self, parser: FilesLoader) -> None:
        pass

    @staticmethod
    def parse_text(parser: FilesLoader):
        files = []

        if parser.encoding_isConsistent:
            for i in parser.file_names:
                with open(
                    str(i),
                    "r",
                    encoding=list(parser.files_and_encoding.values())[0]["encoding"],
                ) as f:
                    doc = f.read()
                    doc = doc.replace("\n", " ")
                    doc = doc.replace("\r", " ")
                    doc = re.sub(r"[^ \nA-Za-z0-9Ã€-Ã–Ã˜-Ã¶Ã¸-Ã¿Ð€-Ó¿/]+", "", doc)
                files.append(str(doc))
        else:
            for i in parser.file_names:
                with open(
                    str(i), "r", encoding=parser.files_and_encoding[i]["encoding"]
                ) as f:
                    doc = " ".join(f.readlines())
                    doc = doc.replace("\n", " ")
                    doc = doc.replace("\r", " ")
                    doc = re.sub(r"[^ \nA-Za-z0-9Ã€-Ã–Ã˜-Ã¶Ã¸-Ã¿Ð€-Ó¿/]+", "", doc)
                files.append(str(doc))
        return files


if __name__ == "__main__":
    with open(
        "D:\Documents\OneDrive\Python\ChromaDB\config\database_settings.yml", "r"
    ) as f:
        config = yaml.safe_load(f)
    text_parser = FilesLoader(config, extension="*.txt")
    print(
        "\nTextParser config:\n",
        text_parser.config,
        "\n\n",
        "File extension:\n",
        text_parser.extension,
        "\n\n",
        "Files directory:\n",
        text_parser.dir_path,
        "\n\n",
        "File names:\n",
        text_parser.file_names,
        "\n\n",
        "Files and their encodings:\n",
        text_parser.files_and_encoding,
        "\n\n",
        "Is the encoding of all files consistent?",
        text_parser.encoding_isConsistent,
        "\n\n",
    )

    encoded_text_parsed = EncodedTextFilesParser.parse_text(text_parser)
    print(
        str(encoded_text_parsed[6]),
        "\n\nLength of text in characters:",
        len(encoded_text_parsed[2]),
        type(encoded_text_parsed[2]),
    )
