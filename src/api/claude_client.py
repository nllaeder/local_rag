"""
Claude API client for generating responses in the RAG system.
Only external API dependency in the architecture.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import asyncio
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class ClaudeClient:
    """Client for interacting with Claude API"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Claude client
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Claude client initialized with model: {model}")
    
    async def generate_response(
        self, 
        user_message: str, 
        context: str, 
        chat_history: List[Dict[str, Any]] = None,
        max_tokens: int = 1000
    ) -> Tuple[str, Optional[int]]:
        """
        Generate response using Claude with RAG context
        
        Args:
            user_message: User's question/message
            context: Retrieved document context
            chat_history: Previous chat messages for context
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (response_text, tokens_used)
        """
        try:
            # Build system prompt with context
            system_prompt = self._build_system_prompt(context)
            
            # Build conversation messages
            messages = self._build_messages(user_message, chat_history)
            
            # Make API call (run in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages
                )
            )
            
            # Extract response and usage info
            response_text = response.content[0].text
            tokens_used = getattr(response.usage, 'output_tokens', None)
            
            logger.info(f"Generated response with {tokens_used} tokens")
            
            return response_text, tokens_used
            
        except Exception as e:
            logger.error(f"Error generating Claude response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}", None
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with document context"""
        return f"""You are a helpful AI assistant with access to document context. Use the provided context to answer questions accurately and cite sources when possible.

Context from documents:
{context}

Guidelines:
1. Answer based on the provided context when relevant
2. If the context doesn't contain enough information, say so clearly
3. Cite sources by mentioning the document name and page number
4. Be concise but comprehensive
5. If asked about information not in the context, explain that you don't have that information in the current documents
"""
    
    def _build_messages(self, user_message: str, chat_history: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Build message list for Claude API"""
        messages = []
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history[-10:]:  # Last 10 messages for context
                if msg['role'] in ['user', 'assistant']:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
        
        # Add current user message
        messages.append({
            'role': 'user',
            'content': user_message
        })
        
        return messages
    
    async def generate_simple_response(self, message: str, max_tokens: int = 500) -> Tuple[str, Optional[int]]:
        """
        Generate simple response without RAG context
        
        Args:
            message: User message
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (response_text, tokens_used)
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": message}]
                )
            )
            
            response_text = response.content[0].text
            tokens_used = getattr(response.usage, 'output_tokens', None)
            
            return response_text, tokens_used
            
        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}", None
    
    async def summarize_document_chunk(self, chunk_text: str, max_tokens: int = 200) -> str:
        """
        Generate summary of a document chunk
        
        Args:
            chunk_text: Text chunk to summarize
            max_tokens: Maximum tokens in summary
            
        Returns:
            Summary text
        """
        prompt = f"Please provide a concise summary of the following text:\n\n{chunk_text}"
        
        response, _ = await self.generate_simple_response(prompt, max_tokens)
        return response
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the Claude model being used"""
        return {
            "model": self.model,
            "provider": "Anthropic",
            "type": "API"
        }