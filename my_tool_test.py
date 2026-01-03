import unittest
from unittest.mock import patch, MagicMock
import os
from my_tool import ask

class TestMyTool(unittest.TestCase):

    @patch('google.generativeai.GenerativeModel')
    def test_ask(self, mock_generative_model):
        # Set a dummy API key for the test
        os.environ["GOOGLE_API_KEY"] = "test_key"

        # Create a mock response
        mock_response = MagicMock()
        mock_response.text = "The answer is 42."

        # Configure the mock model to return the mock response
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model_instance

        # Call the function to be tested
        response = ask("What is the meaning of life?")

        # Assert that the mock was called with the correct question
        mock_model_instance.generate_content.assert_called_once_with("What is the meaning of life?")

        # Assert that the function returns the correct text
        self.assertEqual(response, "The answer is 42.")

if __name__ == '__main__':
    unittest.main()