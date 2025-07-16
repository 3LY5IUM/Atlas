from langchain_google_genai import ChatGoogleGenerativeAI
from .config import Config

from typing import List, Dict


from langchain_core.messages import HumanMessage, SystemMessage

import base64
from PIL import Image
import io


from langchain.prompts import ChatPromptTemplate







config = Config()
llm = ChatGoogleGenerativeAI(
        model=Config.CHAT_MODEL,  # e.g., "gemini-1.5-flash", "gemini-pro-vision"
        api_key=config.GEMINI_API_KEY,
        temperature=0.7,  # Optional: adjust as needed
        max_tokens=None,  # Optional: set token limit
        timeout=None,     # Optional: set timeout
        max_retries=2,    # Optional: set retry attempts
        )



def get_respo(
        query: str,
        results: List[Dict],
        chat_history: List[Dict[str,str]] ) -> str:
    # generate the response to a user query.
        try:
            # retrives relevant documents.

            if not results:
                return "Sorry Sir but I could not find any relevant information in the upladed data to answer this query."

            # build context

            context_parts = []
            image_contents = []


            for i, doc in enumerate(results):
                context_part = f" Document {i+1} (source: {doc['metadata'].get('source', 'who knows')}):\n"

                # image
                if doc['metadata'].get('content_type') == 'image':
                    context_part += f"Image Description: {doc['content']}\n"
                    # Store image data for potential use
                    if doc['metadata'].get('image_data'):
                        image_contents.append({
                            'data': doc['metadata']['image_data'],
                            'description': doc['metadata'].get('image_desc', '')
                            })
                # table
                elif doc['metadata'].get('content_type') == 'table':
                # handel table conent
                    context_part += f"Table content: {doc['content']}\n"
                    if doc['metadata'].get('html_content'):
                        context_part += f"Table HTML: {doc['metadata']['html_content']}\n"

                # text
                else:
                    context_part += f"Content: {doc['content']}\n"


                # add the element to context_parts.
                context_parts.append(context_part)

            # combine all documents into a single string with the record seperator as "\n".
            full_context = "\n".join(context_parts)


            # langchain messages to define the human and system message.
            prompt = f"""Based on the following context from the uploaded documents, please answer the user's question.

Context:
{full_context}

User Question: {query}

Please provide a comprehensive answer based on the context above. If the context includes information from images or tables, make sure to incorporate that information in your response."""


            if chat_history:
                history_text = "\n".join([
                    # we will use the for loop after the fstring. this is list comprehension.
                    f"{msg['role']}: {msg['content']}" 
                    for msg in chat_history[-5:]
                    ])

                # inside the if statement coz the history could be empty
                prompt = f"Previous conversation: \n{history_text}\n\n{prompt}"


            messages = [
                    SystemMessage(
                        content="""You help users understand and analyze documents by answering questions based on the provided context.

            When answering:
            1. Use the provided context from the documents to answer questions accurately
            2. If context includes images, refer to their descriptions when relevant
            3. For tables, use the structured HTML content when available
            4. Be concise but comprehensive in your responses
            5. If you cannot find relevant information in the context, say so clearly
            6. Always cite which part of the document you're referencing when possible
            """
            ),
                    HumanMessage(
                        content=prompt
                        )
                    ] 


            respo = llm.invoke(messages)
            return respo.content

        except Exception as e:
            return f"Sorry Sir, but there is an error while processing the questoins through the llm: {str(e)}"

def analyze_image_with_query(self, image_base64: str, query: str) -> str:
    """Analyze a specific image with a user query"""
    try:
    # Create prompt for specific query
        question = f"""Please analyze this image and answer the following question: {query}

        Provide a detailed response based on what you can see in the image."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{question}"),
            ("human", [
                {
                    "type": "text",
                    "text": "Please analyze this image based on the question above."
                    },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{image_data}"
                        }
                    }
                ])
            ])

        # Generate response with image
        chain = prompt | llm  
        response = chain.invoke(
                {
                    "image_data": image_base64,
                    "question": question,
                    }
                )

        return response.content

    except Exception as e:
        return f"Sorry sir but there was an Error analyzing image for this query: {str(e)}"



















