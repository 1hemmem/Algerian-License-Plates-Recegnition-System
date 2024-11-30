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

        Parameters:
        - image_path: Path to the image file
        - prompt: Instruction for the model on how to extract text

        Returns:
        - Extracted text from the license plate image
        """

        if prompt is None:
            prompt = """Extract the numbers from this license plate, 
            the response should include only the result in the format: xxxxx xxx xx, Where all the x values are numbers"""
        if os.path.exists(image_path):
            # time.sleep(120)
            sample = self.prep_image(image_path)

            model = genai.GenerativeModel(model_name=self.model_name)

            # Try to call the model and handle quota-related exceptions
            while True:
                try:
                    response = model.generate_content([sample, prompt])
                    text_content = response.candidates[0].content.parts[0].text
                    logger.info("Extracted text successfully.")
                    return text_content

                except google.api_core.exceptions.ResourceExhausted:
                    time.sleep(120)
                    logger.warning("Quota exceeded. Retrying after delay.")
                    time.sleep(self.retry_delay)
                except Exception as e:
                    time.sleep(120)
                    logger.error(f"Failed to extract text: {e}")
                    raise
        else:
            logger.warning(f"File: {image_path} Does not exist yachkopi")
