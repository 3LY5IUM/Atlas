# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Config

from typing import List, Dict, Any

# converting base64 to pass to the vision model.
import base64

# for opeing the the image
from PIL import Image
import io

# this is a python class that will have instances with atributes like config.
class PDF_processor:
    def __init__(self):
        self.config = Config()

        self.vision_model = ChatGoogleGenerativeAI(
                model=self.config.VISION_MODEL,  # e.g., "gemini-1.5-flash", "gemini-pro-vision"
                api_key=self.config.GEMINI_API_KEY,
                temperature=0.7,  # Optional: adjust as needed
                max_tokens=None,  # Optional: set token limit
                timeout=None,     # Optional: set timeout
                max_retries=2,    # Optional: set retry attempts
                )


# this function uses typing library to use uppercase annotations like List and not list eventhough you could probolbally use lowercase stff as well.
    def process_pdf(self, pdf_path: str) -> List[ Dict[str, Any] ]:
        """
        this is suppose to extract images, tables and text from pdf
        """
        # using the try block so that if an error occur the program doesn't crashes and instead we could handle the error.
        try:
            loader = UnstructuredPDFLoader(
                    file_path=pdf_path,
                    strategy="hi_res",  # High resolution for better image/table extraction
                    infer_table_structure=True,  # Extract table structure
                    extract_images_in_pdf=True,  # Extract images
                    extract_image_block_types=["Image", "Table"],  # Extract both images and tables as images
                    chunking_strategy="by_title",  # Chunk by document structure
                    max_characters=self.config.CHUNK_SIZE,
                    overlap=self.config.CHUNK_OVERLAP
                    )
            elements = loader.lazy_load()

            # making a list that will have dictuionaries i.e. key value pairs in it. 
            # this would not be much usefull but as we are adding image_description as well we could just make this new list with all the stuff we need from the extracted data.
            processed_elements = []

            for i, element in enumerate(elements):
                processed_element = {
                        "id": f"element_{i}",
                        # "type": doc.metadata.get("category", "unknown") get is used to retrive value of associated keys.
                        "type": element.metadata.get("category", "unknown"),
                        "content": str(element.page_content),
                        "metadata": element.metadata,
                        "source": pdf_path
                        }

                if element.metadata.get("category") == "Table":
                    # storing the html_content 
                    # checking if there is text_as_html attribute.
                    # use dict["key"] when you are certain that key exist and want an error if it doesn't while using the get() allows you to enter a default value.
                    if "text_as_html" in element.metadata and element.metadata["text_as_html"]:
                        processed_element["html_content"] = element.metadata.get("text_as_html")
                        # no real value... just use type.
                    processed_element["content_type"] = "table"
                elif element.metadata.get("category") == "Image":
                    processed_element["content_type"] = "image"

                    if "image_base64" in element.metadata and element.metadata["image_base64"]:
                        image_as_base64 = element.metadata.get("image_base64")
                        # kida not usefull but let it be for the safer side ig.
                        if image_as_base64:
                            processed_element["image_data"] = image_as_base64
                            # processed_element["content_type"] = "image"

                            # using gimini vision to get a summry of image and storing it in.
                            image_desc = self._analyze_image(image_as_base64)
                            processed_element["image_desc"] = image_desc
                            processed_element["content"] = f"Image: {image_desc}"

                else:
                        # regular text
                        processed_element["content_type"] = "text"

                processed_elements.append(processed_element)

            return processed_elements

        except Exception as shit:
            raise Exception(f"I guess i am an illiterate coz i cant read {pdf_path}: {str(shit)}")


    def _analyze_image(self, image_base64: str) -> str:
        """ gets a summary and tries to analyze the image """
        try:
            # decode the image from base64.
            image_data = base64.b64decode(image_base64)
            # ByteIO is used for in in memory data stream.
            image = Image.open(io.BytesIO(image_data))

            # resize the image if it is too large



            # Image.size if a PIL method that return a tupple with (hori, ver) so that is why we are checking both..
            if image.size[0] > self.config.MAX_IMAGE_SIZE[0] or image.size[1] > self.config.MAX_IMAGE_SIZE[1]:
                image.thumbnail(self.config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

            prompt = """Analyze this image and provide a detailed description. Include:
            1. What the image shows (objects, people, scenes, etc.)
            2. Any text visible in the image
            3. Important details that might be relevant for document understanding
            4. If it's a chart, graph, or table, describe the data it contains

            Provide a comprehensive description that contains all the values, data and key findings from the image.
            """


            messages = [ HumanMessage( 
                                      content = [
                                          {
                                              "type": "text",
                                              "text": prompt
                                              },
                                          {
                                              "type": "image_url",
                                              "image_url": {
                                                  "url": f"data:image/jpeg;base64,{image_base64}",
                                                  "detail": "high"  # or "low" for faster processing
                                                  }
                                              }

                                          ]
                                      ),
                        SystemMessage("you are an image analyzing assistant, analyze all images with atmost accuracy to retrive all information from it.")
                        ]

            # generating a response.
            respo = self.vision_model.invoke([messages])


            if isinstance(respo, str):
                return respo.content
            else:
                raise ValueError("Expected a string in respo.content, got: {}".format(type(respo.content)))



        except Exception as e:
            print(f"Error analyzing image with Gemini: {str(e)}")
            return "Image could not be analyzed for image description."

















