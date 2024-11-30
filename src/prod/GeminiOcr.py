import os
import google.generativeai as genai
import google
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiOCR:
    def __init__(self, api_key: str, model_name: str, retry_delay: int = 30):
        """
        Initialize the GeminiOCR with an API key and model name.

        Parameters:
        - api_key: API key for Google Generative AI
        - model_name: Name of the model to use
        - retry_delay: Time in seconds to wait before retrying a failed request due to quota limits
        """
        self.api_key = api_key
        self.model_name = model_name
        self.retry_delay = retry_delay
        genai.configure(api_key=self.api_key)

    def prep_image(self, image_path: str, max_retries=3):
        """
        Uploads the image file to the generative AI platform with retries.

        Parameters:
        - image_path: Path to the image file
        - max_retries: Maximum number of retries if the upload fails

        Returns:
        - Uploaded file object with URI
        """
        retries = 0
        while retries < max_retries:
            try:
                sample_file = genai.upload_file(
                    path=image_path, display_name="License plate"
                )
                logger.info(
                    f"Uploaded file '{sample_file.display_name}' as '{sample_file.uri}'"
                )
                return sample_file
            except Exception as e:
                retries += 1
                logger.error(
                    f"Failed to upload image: {e}. Retry {retries}/{max_retries}"
                )
                time.sleep(5)  # Wait before retrying
                if retries == max_retries:
                    raise

    async def extract_text_from_image(self, image_path: str, prompt: str = None):
        """
        Extracts text from the image by calling the generative AI API.
        """
        if not os.path.exists(image_path):
            logger.warning(f"File not found: {image_path}")
            return None

        try:
            if prompt is None:
                prompt = """Extract the numbers from this license plate, 
                the response should include only the result in the format: xxxxx xxx xx, Where all the x values are numbers"""
            
            sample = self.prep_image(image_path)
            if not sample:
                logger.error("Failed to prepare image")
                return None

            model = genai.GenerativeModel(model_name=self.model_name)

            while True:
                try:
                    response = model.generate_content([sample, prompt])
                    if not response.candidates:
                        logger.warning("No response candidates received")
                        return None
                    
                    text_content = response.candidates[0].content.parts[0].text
                    logger.info(f"Extracted text: {text_content}")
                    return text_content.strip()

                except google.api_core.exceptions.ResourceExhausted:
                    logger.warning("Quota exceeded. Retrying after delay.")
                    time.sleep(self.retry_delay)
                except Exception as e:
                    logger.error(f"Failed to extract text: {e}")
                    return None

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
