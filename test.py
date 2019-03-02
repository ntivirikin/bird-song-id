import unittest
import os
from preprocess import resample_wav

class TestCases(unittest.TestCase):

    # Check if files were converted to desire rate
    def test_resample_wav(self):
        result = resample_wav(os.getcwd() + '/data/input/Black-billedPeppershrike5890.mp3', os.getcwd() + '/data/output/Black-billedPeppershrike5890.wav', 16000)
        self.assertTrue(os.path.exists(os.getcwd() + '/data/output/Black-billedPeppershrike5890.wav'))
        self.assertEqual(result[1], '16000')