# import google.generativeai as genai
from langchain_community.document_loaders import UnstructuredPDFLoader

from .config import Config

from typing import List, Dict, Any

import hashlib
import os

# this is a python class that will have instances with atributes like config.
class PDF_processor:
    def __init__(self):
        self.config = Config()
        self._seen: set[str] = set()


    def normalize_path(self, file_path: str) -> str:
        """Normalize file path for consistent comparison"""
        # Convert to absolute path and normalize
        abs_path = os.path.abspath(file_path)
        # Normalize path separators and remove redundant elements
        normalized = os.path.normpath(abs_path)
        return normalized
        
    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash of file content for change detection"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error hashing file {file_path}: {str(e)}")
            return ""

    def is_file_already_processed(self, file_path: str, file_hash: str, vector_store) -> bool:
        """Check if file with same content hash exists in vector store"""
        try:
             # Normalize the file path for consistent comparison
            print(f"With hash: {file_hash}")
            # Query ChromaDB for documents with this file path and hash
            results = vector_store.get(
                where={"file_hash": file_hash},
                include=["metadatas"]
            )
            print(f"Found {len(results.get('metadatas', []))} documents with matching source")
            if results.get('metadatas'):
                print(f"File already processed with same hash: {file_hash}")
                return True

            return False

        except Exception as e:
            print(f"Error checking if file processed: {str(e)}")
            return False



    def remove_old_embeddings(self, file_path: str, vector_store):
        """Remove old embeddings for a file when content has changed"""
        try:
            normalized_path = self.normalize_path(file_path)
            # Get all documents with this source path
            results = vector_store.get(
                where={"source": normalized_path},
                include=["metadatas"]
            )
            
            if results.get('ids'):
                # Delete old embeddings
                vector_store.delete(ids=results['ids'])
                print(f"Removed {len(results['ids'])} old embeddings for {file_path}")
        except Exception as e:
            print(f"Error removing old embeddings: {str(e)}")

# this function uses typing library to use uppercase annotations like List and not list eventhough you could probolbally use lowercase stff as well.
    def process_pdf(self, pdf_path: str, vector_store=None) -> List[ Dict[str, Any] ]:
        """
        this is suppose to extract images, tables and text from pdf
        """
        if pdf_path in self._seen:
            # nothing new to embed
            return []
        # using the try block so that if an error occur the program doesn't crashes and instead we could handle the error.
        try:
             # Normalize the file path
            normalized_path = self.normalize_path(pdf_path)
             # Calculate file hash for change detection
            file_hash = self.get_file_hash(pdf_path)
            if not file_hash:
                raise Exception(f"Could not generate hash for {pdf_path}")

            print(f"Processing file: {normalized_path}")
            print(f"File hash: {file_hash}")



            # check if file is already processed with same content
            if vector_store and self.is_file_already_processed(normalized_path, file_hash, vector_store):
                print(f"File {pdf_path} already processed with same content. Skipping...")
                return []

            if vector_store:
                print(f"Removing old embeddings for {normalized_path}")
                self.remove_old_embeddings(pdf_path, vector_store)



            loader = UnstructuredPDFLoader(
                    file_path=pdf_path,
                    strategy="fast",  # High resolution for better image/table extraction
                    infer_table_structure=False,  # Extract table structure
                    extract_images_in_pdf=False,  # Extract images
                    # extract_image_block_types=["Image", "Table"],  # Extract both images and tables as images
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
                        "file_hash": file_hash,
                        "source": normalized_path,
                        }

                # regular text
                processed_element["content_type"] = "text"

                processed_elements.append(processed_element)

            return processed_elements

        except Exception as shit:
            raise Exception(f"I guess i am an illiterate coz i cant read {pdf_path}: {str(shit)}")


    # def _analyze_image(self, image_base64: str) -> str:
    #     """ gets a summary and tries to analyze the image """
    #     try:
    #         # decode the image from base64.
    #         image_data = base64.b64decode(image_base64)
    #         # ByteIO is used for in in memory data stream.
    #         image = Image.open(io.BytesIO(image_data))
    #
    #
    #
    #
    #         # Image.size if a PIL method that return a tupple with (hori, ver) so that is why we are checking both..
    #         if image.size[0] > self.config.MAX_IMAGE_SIZE[0] or image.size[1] > self.config.MAX_IMAGE_SIZE[1]:
    #             image.thumbnail(self.config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    #
    #         prompt = """Analyze this image and provide a detailed description. Include:
    #         1. What the image shows (objects, people, scenes, etc.)
    #         2. Any text visible in the image
    #         3. Important details that might be relevant for document understanding
    #         4. If it's a chart, graph, or table, describe the data it contains
    #
    #         Provide a comprehensive description that contains all the values, data and key findings from the image.
    #         """
    #
    #
    #         messages = [ HumanMessage( 
    #                                   content = [
    #                                       {
    #                                           "type": "text",
    #                                           "text": prompt
    #                                           },
    #                                       {
    #                                           "type": "image_url",
    #                                           "image_url": {
    #                                               "url": f"data:image/jpeg;base64,{image_base64}",
    #                                               "detail": "high"  # or "low" for faster processing
    #                                               }
    #                                           }
    #
    #                                       ]
    #                                   ),
    #                     SystemMessage("you are an image analyzing assistant, analyze all images with atmost accuracy to retrive all information from it.")
    #                     ]
    #
    #         # generating a response.
    #         respo = self.vision_model.invoke([messages])
    #
    #
    #         if isinstance(respo, str):
    #             return respo.content
    #         else:
    #             raise ValueError("Expected a string in respo.content, got: {}".format(type(respo.content)))
    #
        #
        #
        # except Exception as e:
        #     print(f"Error analyzing image with Gemini: {str(e)}")
        #     return "Image could not be analyzed for image description."
        #
        #















