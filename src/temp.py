# Fixed src/chat.py - Chat Handler with Gemini

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from .config import Config
from typing import List, Dict
import base64
from PIL import Image
import io

# Initialize LLM at module level
config = Config()
llm = ChatGoogleGenerativeAI(
    model=config.CHAT_MODEL,
    api_key=config.GEMINI_API_KEY,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def get_respo(query: str, vector_store, chat_history: List[Dict[str, str]]) -> str:
    """Generate response to user query using retrieved context"""
    try:
        # Retrieve relevant documents
        relevant_docs = vector_store.query(query)

        if not relevant_docs:
            return "Sorry Sir but I could not find any relevant information in the uploaded data to answer this query."

        # Build context
        context_parts = []
        image_contents = []

        for i, doc in enumerate(relevant_docs):
            context_part = f"Document {i+1} (source: {doc['metadata'].get('source', 'unknown')}):\n"

            # Handle different content types
            if doc['metadata'].get('content_type') == 'image':
                context_part += f"Image Description: {doc['content']}\n"
                # Store image data for potential use
                if doc['metadata'].get('image_data'):
                    image_contents.append({
                        'data': doc['metadata']['image_data'],
                        'description': doc['metadata'].get('image_desc', '')
                    })
            elif doc['metadata'].get('content_type') == 'table':
                context_part += f"Table content: {doc['content']}\n"
                if doc['metadata'].get('html_content'):
                    context_part += f"Table HTML: {doc['metadata']['html_content']}\n"
            else:
                context_part += f"Content: {doc['content']}\n"

            context_parts.append(context_part)

        # Combine all documents into a single string
        full_context = "\n".join(context_parts)

        # Create prompt
        prompt = f"""Based on the following context from the uploaded documents, please answer the user's question.

Context:
{full_context}

User Question: {query}

Please provide a comprehensive answer based on the context above. If the context includes information from images or tables, make sure to incorporate that information in your response."""

        # Add chat history if available
        if chat_history:
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in chat_history[-5:]  # Last 5 messages
            ])
            prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"

        # Create messages
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
7. Address the user as 'sir' and be formal and respectful
"""
            ),
            HumanMessage(content=prompt)
        ]

        # Generate response
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"Sorry Sir, but there is an error while processing the question through the LLM: {str(e)}"

def analyze_image_with_query(image_base64: str, query: str) -> str:
    """Analyze a specific image with a user query"""
    try:
        # Decode and prepare image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Resize if needed
        config = Config()
        if image.size[0] > config.MAX_IMAGE_SIZE[0] or image.size[1] > config.MAX_IMAGE_SIZE[1]:
            image.thumbnail(config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

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
        response = chain.invoke({
            "image_data": image_base64,
            "question": question,
        })

        return response.content

    except Exception as e:
        return f"Sorry sir but there was an error analyzing image for this query: {str(e)}"
